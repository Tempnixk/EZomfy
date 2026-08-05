"""워크플로우 JSON + 매핑 TOML을 읽어 파라미터를 주입한다.

워크플로우는 코드가 아니라 데이터다 (CLAUDE.md 6장). ComfyUI에서
"Save (API Format)"으로 내보낸 JSON을 그대로 workflows/에 두고, 이 모듈은
매핑 TOML이 가리키는 노드의 inputs만 교체한다. 노드 그래프 자체를
재구성하지 않으며, 노드 ID를 코드에 하드코딩하지 않는다.
"""
from __future__ import annotations

import copy
import json
import tomllib
from typing import Any

from styleforge.config import settings


class WorkflowLoadError(RuntimeError):
    """워크플로우 JSON 또는 매핑 TOML 로드/적용 중 발생한 오류."""


def load_workflow(name: str) -> dict[str, Any]:
    workflow_path = settings.workflows_dir / f"{name}.json"
    if not workflow_path.is_file():
        raise WorkflowLoadError(f"Workflow file not found: {workflow_path}")

    with workflow_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_mapping(name: str) -> dict[str, dict[str, str]]:
    mapping_path = settings.workflows_dir / "mappings" / f"{name}.toml"
    if not mapping_path.is_file():
        raise WorkflowLoadError(f"Mapping file not found: {mapping_path}")

    with mapping_path.open("rb") as f:
        raw = tomllib.load(f)

    params = raw.get("params", {})
    if not params:
        raise WorkflowLoadError(f"No [params.*] entries defined in {mapping_path}")

    return params


def apply_params(
    workflow: dict[str, Any],
    mapping: dict[str, dict[str, str]],
    values: dict[str, Any],
) -> dict[str, Any]:
    """values의 각 파라미터를 mapping이 가리키는 노드 inputs에 주입한 새 워크플로우를 반환한다."""
    result = copy.deepcopy(workflow)

    for param_name, value in values.items():
        target = mapping.get(param_name)
        if target is None:
            raise WorkflowLoadError(f"Unknown parameter '{param_name}' — not defined in mapping TOML")

        node_id, field = target["node"], target["field"]
        if node_id not in result:
            raise WorkflowLoadError(f"Mapping references node '{node_id}' which does not exist in workflow")

        result[node_id]["inputs"][field] = value

    return result


def load_and_apply(name: str, values: dict[str, Any]) -> dict[str, Any]:
    """workflows/{name}.json + workflows/mappings/{name}.toml을 읽어 파라미터를 주입한다."""
    workflow = load_workflow(name)
    mapping = load_mapping(name)
    return apply_params(workflow, mapping, values)
