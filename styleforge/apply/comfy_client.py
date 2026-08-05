"""ComfyUI API 클라이언트 — 워크플로우 제출부터 결과 이미지 다운로드까지 담당한다.

ComfyUI와의 통신은 반드시 이 모듈을 거친다 (CLAUDE.md 8장). 다른 모듈에서
ComfyUI에 직접 HTTP/WebSocket 요청을 보내지 않는다.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
import websocket


class ComfyClientError(RuntimeError):
    """ComfyUI와의 통신 또는 워크플로우 실행 중 발생한 오류."""


@dataclass
class ProgressUpdate:
    value: int
    max: int


class ComfyClient:
    """ComfyUI HTTP/WebSocket API의 유일한 진입점."""

    def __init__(self, base_url: str, client_id: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())

    def upload_image(self, image_path: Path, *, overwrite: bool = True) -> str:
        """이미지를 ComfyUI 서버에 업로드하고 서버상의 파일명을 반환한다."""
        if not image_path.is_file():
            raise ComfyClientError(f"Image file not found: {image_path}")

        try:
            with image_path.open("rb") as f:
                files = {"image": (image_path.name, f, "image/png")}
                data = {"overwrite": "true" if overwrite else "false"}
                response = requests.post(f"{self.base_url}/upload/image", files=files, data=data, timeout=30)
        except requests.exceptions.RequestException as exc:
            raise ComfyClientError(f"Failed to reach ComfyUI at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise ComfyClientError(f"Image upload failed ({response.status_code}): {response.text}")

        return response.json()["name"]

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """워크플로우 JSON(API 포맷)을 제출하고 prompt_id를 반환한다."""
        payload = {"prompt": workflow, "client_id": self.client_id}

        try:
            response = requests.post(f"{self.base_url}/prompt", json=payload, timeout=30)
        except requests.exceptions.RequestException as exc:
            raise ComfyClientError(f"Failed to reach ComfyUI at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise ComfyClientError(f"Prompt submission failed ({response.status_code}): {response.text}")

        body = response.json()
        if body.get("error"):
            raise ComfyClientError(f"Workflow rejected by ComfyUI: {body['error']}")

        return body["prompt_id"]

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
        except requests.exceptions.RequestException as exc:
            raise ComfyClientError(f"Failed to reach ComfyUI at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise ComfyClientError(f"History fetch failed ({response.status_code}): {response.text}")

        return response.json()

    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}

        try:
            response = requests.get(f"{self.base_url}/view", params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            raise ComfyClientError(f"Failed to reach ComfyUI at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise ComfyClientError(f"Image download failed ({response.status_code}): {response.text}")

        return response.content

    def wait_for_completion(
        self,
        prompt_id: str,
        *,
        on_progress: Callable[[ProgressUpdate], None] | None = None,
        timeout: float = 600.0,
    ) -> None:
        """WebSocket으로 진행률을 수신하며 해당 prompt_id의 실행 완료를 기다린다."""
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")

        try:
            ws = websocket.create_connection(f"{ws_url}/ws?clientId={self.client_id}", timeout=timeout)
        except (OSError, websocket.WebSocketException) as exc:
            raise ComfyClientError(f"Failed to open WebSocket to {ws_url}: {exc}") from exc

        try:
            while True:
                try:
                    raw = ws.recv()
                except (OSError, websocket.WebSocketException) as exc:
                    raise ComfyClientError(f"WebSocket connection lost while waiting for {prompt_id}: {exc}") from exc

                if not isinstance(raw, str):
                    continue  # 바이너리 프리뷰 프레임은 사용하지 않는다

                message = json.loads(raw)
                msg_type = message.get("type")
                data = message.get("data", {})

                if msg_type == "progress" and on_progress is not None:
                    on_progress(ProgressUpdate(value=data.get("value", 0), max=data.get("max", 0)))

                elif msg_type == "executing" and data.get("prompt_id") == prompt_id and data.get("node") is None:
                    return

                elif msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                    raise ComfyClientError(f"Workflow execution failed for prompt_id={prompt_id}: {data}")
        finally:
            ws.close()

    def run_workflow(
        self,
        workflow: dict[str, Any],
        output_dir: Path,
        *,
        on_progress: Callable[[ProgressUpdate], None] | None = None,
    ) -> list[Path]:
        """워크플로우를 제출하고 완료를 기다린 뒤 결과 이미지를 output_dir에 저장한다."""
        prompt_id = self.queue_prompt(workflow)
        self.wait_for_completion(prompt_id, on_progress=on_progress)

        history = self.get_history(prompt_id)
        if prompt_id not in history:
            raise ComfyClientError(f"No history entry for prompt_id={prompt_id}")

        outputs = history[prompt_id].get("outputs", {})
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        for node_output in outputs.values():
            for image_info in node_output.get("images", []):
                image_bytes = self.get_image(
                    image_info["filename"],
                    image_info.get("subfolder", ""),
                    image_info.get("type", "output"),
                )
                save_path = output_dir / image_info["filename"]
                save_path.write_bytes(image_bytes)
                saved_paths.append(save_path)

        if not saved_paths:
            raise ComfyClientError(f"Workflow completed but produced no output images (prompt_id={prompt_id})")

        return saved_paths
