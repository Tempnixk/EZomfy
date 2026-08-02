# CLAUDE.md

이 파일은 Claude Code가 이 저장소에서 작업할 때 참조하는 프로젝트 컨텍스트입니다.

---

## 1. 프로젝트 개요

**art-pipeline-tool** — 아트 직군을 위한 사내 아트 생산성 통합 툴 (포트폴리오 프로젝트)

ComfyUI를 렌더링 백엔드로 사용하여, 기획자·아티스트가 노드 그래프를 다루지 않고도
프로젝트 화풍에 맞는 컨셉 시안을 생성할 수 있게 하는 웹 기반 사내 툴입니다.

핵심 가치 제안:
- 아티스트는 ComfyUI 노드를 몰라도 된다. 스케치 업로드 + 프리셋 선택 + 생성 버튼이 전부다.
- 생성 결과는 항상 메타데이터(워크플로우, 파라미터, 시드)와 함께 저장되어 재현 가능하다.
- 새 워크플로우 추가는 코드 수정 없이 JSON + 프리셋 등록만으로 끝난다.

### 주요 기능 영역
1. 스케치 → 화풍 적용 컨셉 시안 생성 (ControlNet + 화풍 LoRA)
2. 화풍 LoRA 학습 파이프라인 (데이터 수집 → 전처리 → 학습 → 검증)
3. UE5 블록아웃 스크린샷 → 배경 컨셉 생성 (depth/canny ControlNet)
4. 결과물 갤러리 + 사용자 피드백(태그·좋아요) 수집
5. 작업 큐 관리 및 리소스(VRAM) 운영

---

## 2. 핵심 제약 조건 (반드시 준수)

> 아래 제약은 이 프로젝트의 모든 설계 결정을 지배합니다. 코드 작성 전 항상 확인하세요.

### VRAM 8GB (RTX 3060 Ti)
- **동시 생성 작업은 반드시 1개로 제한한다.** `queue_manager.py`가 이를 강제한다.
  병렬 처리 코드를 제안하지 말 것.
- **LoRA 학습과 ComfyUI 서버는 동시에 실행할 수 없다.**
  학습 스크립트는 시작 전 반드시 `scripts/stop_comfy.ps1`을 호출해 VRAM을 회수한다.
- 학습 베이스 모델은 **SD 1.5** (반복 실험 가능), 서빙용 추론은 **SDXL 병행**.
  SDXL 학습 코드는 작성하지 않는다.

### Docker 미사용
- `docker-compose.yml`, `Dockerfile`을 만들지 않는다.
- 프로세스 기동은 `scripts/` 하위의 PowerShell 스크립트로 처리한다.
- 개발 환경은 Windows + 네이티브 실행 기준이다.

### 가상환경 분리
- `server/.venv`와 `training/.venv`는 **별도로 유지한다.** 의존성을 합치지 말 것.
  (kohya_ss와 ComfyUI/FastAPI의 torch 버전 충돌 방지)
- ComfyUI는 프로젝트 외부 경로에 설치되어 있으며, 이 저장소가 관리하지 않는다.

### 경로 하드코딩 금지
- ComfyUI 경로, 모델 경로, 출력 경로는 모두 `.env` → `config.py`를 거쳐 주입한다.
- 새 경로가 필요하면 `.env.example`에도 함께 추가한다.

---

## 3. 기술 스택

| 영역 | 스택 |
|---|---|
| 백엔드 | Python 3.11, FastAPI, uvicorn, SQLModel, SQLite |
| 프런트엔드 | React 18, TypeScript, Vite |
| 생성 백엔드 | ComfyUI (API 모드, 별도 프로세스) |
| 학습 | kohya_ss (sd-scripts), SD 1.5 베이스 |
| 실행 관리 | PowerShell 스크립트 |

---

## 4. 디렉터리 구조

```
art-pipeline-tool/
├── CLAUDE.md
├── README.md                      # 아티스트 대상 온보딩 문서
├── .env.example
├── scripts/
│   ├── run_all.ps1                # ComfyUI + server + web 동시 기동
│   ├── run_server.ps1
│   ├── run_web.ps1
│   └── stop_comfy.ps1             # 학습 전 VRAM 회수
│
├── server/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py              # .env 로드, 전역 설정
│   │   ├── api/
│   │   │   ├── generate.py        # POST /api/generate
│   │   │   ├── jobs.py            # GET /api/jobs, /api/jobs/{id}
│   │   │   ├── gallery.py         # 결과물 목록, 태그·좋아요
│   │   │   └── presets.py         # 스타일 프리셋 CRUD
│   │   ├── core/
│   │   │   ├── comfy_client.py    # ★ ComfyUI API 통신 (심장부)
│   │   │   ├── workflow_loader.py # JSON 템플릿 + 파라미터 주입
│   │   │   └── queue_manager.py   # 작업 큐 (동시 1개)
│   │   ├── models/job.py
│   │   └── db.py
│   ├── workflows/                 # ComfyUI API 포맷 JSON 템플릿
│   │   ├── sketch2concept_sd15.json
│   │   ├── sketch2concept_sdxl.json
│   │   ├── blockout2env.json
│   │   └── character_sheet.json
│   ├── tests/
│   └── pyproject.toml
│
├── web/
│   └── src/
│       ├── pages/{Generate,Gallery,Jobs}.tsx
│       ├── components/
│       ├── api/client.ts
│       └── hooks/useJobProgress.ts
│
├── training/
│   ├── scripts/
│   │   ├── 01_collect.py          # 공공 도메인 이미지 수집
│   │   ├── 02_preprocess.py       # 크롭·리사이즈·캡셔닝
│   │   ├── 03_train.py            # kohya_ss 래퍼
│   │   └── 04_validate.py         # 고정 프롬프트로 비교 이미지 생성
│   ├── configs/                   # 실험별 학습 설정
│   │   ├── exp01_rank8.toml
│   │   └── exp02_rank16.toml
│   └── results/                   # 실험 비교 기록 (md + 이미지)
│
└── docs/
    ├── user-guide.md              # 아티스트용 사용 매뉴얼
    ├── architecture.md            # 시스템 구조도
    └── benchmarks.md              # 8GB 제약 하 모델 선택 근거 실측표
```

---

## 5. ComfyUI 연동 규약

ComfyUI는 `--listen 127.0.0.1 --port 8188`로 기동하며, 기본 주소는 `COMFY_URL` 환경변수로 주입합니다.

### 사용하는 엔드포인트
- `POST /prompt` — 워크플로우 제출. 응답으로 `prompt_id`를 받는다.
- `GET /history/{prompt_id}` — 완료된 작업의 출력 노드별 결과 조회.
- `GET /view?filename=...&subfolder=...&type=output` — 생성 이미지 다운로드.
- `POST /upload/image` — 입력 스케치 업로드.
- `WS /ws?clientId={uuid}` — 진행률 수신.

### WebSocket 메시지 처리
- `type: "progress"` → `data.value` / `data.max`로 진행률 계산
- `type: "executing"` → `data.node`가 `null`이면 해당 `prompt_id` 실행 완료
- `type: "execution_error"` → 작업 실패 처리, 에러 메시지를 job 레코드에 저장

### 워크플로우 JSON 취급 원칙 (중요)
- 워크플로우는 **코드가 아니라 데이터**다. ComfyUI에서 "Save (API Format)"으로 내보낸
  JSON을 `server/workflows/`에 그대로 둔다.
- `workflow_loader.py`는 이 JSON을 로드해 **특정 노드 ID의 inputs 값만 교체**한다.
  (positive prompt, negative prompt, 이미지 파일명, seed, steps, cfg, denoise, lora_strength 등)
- 노드 ID를 코드에 하드코딩하지 말고, 워크플로우별 **파라미터 매핑 정의**를 두어
  `{"positive_prompt": {"node": "6", "field": "text"}}` 형태로 관리한다.
- 새 워크플로우 추가 시 코드 수정 없이 JSON + 매핑 정의 + 프리셋 등록만으로 동작해야 한다.

---

## 6. 코딩 규약

### Python
- 타입 힌트 필수. `from __future__ import annotations` 사용.
- FastAPI 라우터는 얇게 유지하고, 로직은 `core/`에 둔다.
- ComfyUI 통신은 반드시 `comfy_client.py`를 경유한다. 다른 모듈에서 직접 HTTP 호출 금지.
- 예외는 삼키지 말고 로깅 후 상위로 전파한다. 작업 실패는 job 레코드에 사유를 남긴다.

### TypeScript
- `any` 금지. 서버 응답 타입은 `web/src/api/types.ts`에 정의한다.
- API 호출은 `api/client.ts`를 경유한다. 컴포넌트에서 직접 `fetch` 금지.
- 상태 관리는 별도 라이브러리 없이 React 내장 훅으로 처리한다.

### 공통
- 주석과 커밋 메시지는 한국어로 작성한다.
- 사용자에게 노출되는 UI 문구는 한국어로 작성한다.
- 로그 메시지는 영어로 작성한다.

---

## 7. 구현 순서 (Walking Skeleton 우선)

기능을 세로로(백엔드 전체 → 프런트 전체) 자르지 말고, **가로로 관통하는 최소 동작**을 먼저 완성합니다.

- [ ] **Phase 0** — `comfy_client.py`를 단독 스크립트로 검증.
      워크플로우 JSON 제출 → WebSocket 진행률 수신 → 결과 이미지 저장까지 CLI로 확인.
- [ ] **Phase 1** — FastAPI로 래핑. `POST /api/generate` + `GET /api/jobs/{id}` 동작.
- [ ] **Phase 2** — 웹 프런트 최소 화면. 스케치 업로드 → 생성 → 결과 표시 1줄기 완성.
- [ ] **Phase 3** — 갤러리, 프리셋, 큐 모니터 확장. 피드백(태그·좋아요) 수집.
- [ ] **Phase 4** — LoRA 학습 파이프라인 (`training/`). 최소 3회 실험 + 비교 문서화.
- [ ] **Phase 5** — `scripts/run_all.ps1`, README, user-guide, benchmarks 문서 정리.

**Phase 0이 검증되기 전에는 다른 파일을 만들지 마세요.** 이 프로젝트의 모든 기능이
`comfy_client.py` 위에 얹히므로, 여기가 불안정하면 전체가 무너집니다.

---

## 8. 화풍 / 데이터 정책

- 학습 데이터는 **공공 도메인 자료만** 사용한다. (조선 민화·풍속화 — 국립중앙박물관 e뮤지엄,
  국가유산청 공공누리 등 이용 조건이 명시된 출처)
- 데이터 출처와 이용 조건을 `training/results/`에 기록으로 남긴다.
- 특정 상용 IP나 생존 작가의 화풍을 모사하는 데이터는 수집하지 않는다.
- 데모용 프롬프트 소재는 중립적으로 유지한다. (예: 산속 주막, 저잣거리, 호랑이와 사냥꾼)

---

## 9. 하지 말아야 할 것

- Docker 관련 파일 생성
- 동시 생성 작업 병렬화 / SDXL 학습 코드
- 경로 하드코딩 (`D:\ComfyUI` 등을 코드에 직접 기입)
- `server/`와 `training/`의 의존성 통합
- 워크플로우 로직을 Python 코드로 재구현 (ComfyUI JSON을 그대로 쓸 것)
- 요청하지 않은 리팩터링이나 파일 대량 생성 — 한 번에 한 Phase씩 진행

---

## 10. 자주 쓰는 명령

```powershell
# 전체 기동 (ComfyUI + 서버 + 웹)
.\scripts\run_all.ps1

# 서버만
.\scripts\run_server.ps1

# ComfyUI 종료 (학습 전 VRAM 회수)
.\scripts\stop_comfy.ps1

# 학습 실행 (내부에서 stop_comfy 자동 호출)
cd training; .\.venv\Scripts\activate; python scripts\03_train.py --config configs\exp01_rank8.toml

# 서버 테스트
cd server; .\.venv\Scripts\activate; pytest
```
