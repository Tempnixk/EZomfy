"""백그라운드 스레드 + 스레드-세이프 진행률/확인창 브릿지.

tkinter는 메인 스레드 밖에서 위젯을 직접 건드리면 안 된다. `train`처럼 몇
시간 걸리는 작업을 GUI를 멈추지 않고 돌리려면 백그라운드 스레드가 필요한데,
그 스레드 안에서 나오는 진행률 갱신과 (`train`의 경고 확인처럼) 사용자 입력
요청을 큐로 메인 스레드에 넘겨야 한다. 이 모듈이 그 다리 역할만 한다 —
실제 작업 로직은 전혀 모른다.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ProgressEvent:
    payload: Any


@dataclass
class ConfirmRequest:
    messages: list[str]
    event: threading.Event
    result: list[bool]  # 1칸짜리 가변 상자 — 메인 스레드가 응답을 여기 담는다


@dataclass
class DoneEvent:
    result: Any


@dataclass
class ErrorEvent:
    exc: Exception


GuiEvent = ProgressEvent | ConfirmRequest | DoneEvent | ErrorEvent


class BackgroundTask:
    """대상 함수를 백그라운드 스레드에서 실행하고, 진행률/확인/완료/에러를
    큐로 흘려보낸다. 메인 스레드는 `poll()`을 `root.after`로 주기적으로 불러
    큐를 비운다.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[GuiEvent] = queue.Queue()

    def make_progress_callback(self) -> Callable[[Any], None]:
        """runner의 `on_progress`에 그대로 넘길 콜백. 백그라운드 스레드에서 호출된다."""

        def _on_progress(update: Any) -> None:
            self._queue.put(ProgressEvent(update))

        return _on_progress

    def make_confirm_callback(self) -> Callable[[list[str]], bool]:
        """`train`의 `on_warning`에 넘길 콜백.

        메인 스레드가 대화상자 결과를 채워줄 때까지 백그라운드 스레드를
        블록시킨다 — `messagebox`는 메인 스레드에서만 열어야 하기 때문에
        직접 여기서 열 수 없다.
        """

        def _on_warning(messages: list[str]) -> bool:
            event = threading.Event()
            result = [False]
            self._queue.put(ConfirmRequest(messages, event, result))
            event.wait()
            return result[0]

        return _on_warning

    def start(self, target: Callable[[], Any]) -> None:
        def _wrapper() -> None:
            try:
                result = target()
            except Exception as exc:  # 백그라운드 스레드의 예외는 여기서 잡지 않으면
                self._queue.put(ErrorEvent(exc))  # 그냥 사라진다 - GUI로 반드시 넘겨야 함
            else:
                self._queue.put(DoneEvent(result))

        threading.Thread(target=_wrapper, daemon=True).start()

    def poll(self) -> list[GuiEvent]:
        """큐에 쌓인 이벤트를 전부 꺼내 반환한다. 블록하지 않는다."""
        events: list[GuiEvent] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events
