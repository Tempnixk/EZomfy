# CLAUDE.md

이 파일은 Claude Code가 이 저장소에서 작업할 때 참조하는 프로젝트 컨텍스트입니다.

---

## 1. 프로젝트 개요

**StyleForge** — 화풍 학습 · 적용 자동화 CLI 툴

레퍼런스 이미지 폴더 하나를 던지면 화풍 LoRA를 학습하고, 임의의 입력 이미지를
그 화풍으로 변환하는 파이썬 툴입니다. 학습부터 적용, 품질 검증까지 한 파이프라인으로 연결합니다.

### 핵심 명령 (이 3개가 프로젝트의 전부)

```bash
# 1) 폴더 → LoRA 학습
styleforge train --input ./refs/minhwa --name minhwa

# 2) 이미지 → 화풍 변환
styleforge apply --image photo.jpg --style minhwa --strength 0.6

# 3) 파라미터 탐색 + 비교 그리드 생성
styleforge sweep --image photo.jpg --style minhwa
```

### 설계 철학
- **사용자는 denoise/rank/lr을 몰라도 된다.** 기본값으로 쓸 만한 결과가 나와야 한다.
- **모든 실행은 재현 가능하다.** 결과물 옆에 항상 파라미터 메타데이터를 남긴다.
- **튜닝은 감이 아니라 측정이다.** `sweep`이 파라미터 조합을 자동 탐색하고 수치로 평가한다.

---

## 2. 이 프로젝트가 푸는 진짜 문제

> 아래 3가지가 이 툴의 기술적 핵심입니다. 구현 시 항상 의식하세요.

### (1) denoise strength 딜레마
img2img에서 denoise를 낮추면 원본 구조는 지켜지지만 화풍이 입지 않고,
높이면 화풍은 입지만 원본이 사라진다.

**해결 전략:** ControlNet(lineart / canny / depth)으로 구조를 고정한 채 denoise를 높인다.
조절 축은 최소 3개다 — `denoise`, `lora_weight`, `controlnet_strength`.
이 3개의 상호작용을 다루는 것이 `apply` / `sweep`의 존재 이유다.

### (2) 캡셔닝 전략이 화풍 학습 품질을 좌우한다
- 캡션에 스타일 요소를 **남기면** → 트리거 워드로 분리되어 on/off 제어가 가능
- 캡션에서 스타일 요소를 **제거하면** → 스타일이 LoRA 전반에 녹아들어 항상 적용됨

이 프로젝트는 **후자(스타일 요소 제거 + 트리거 워드 1개 부여)** 를 기본 전략으로 한다.

캡션 소스는 두 경로를 모두 지원한다. 둘의 학습 품질 차이를 측정해
`docs/experiments.md`에 비교 기록으로 남기는 것이 이 프로젝트의 실험 항목 중 하나다.

- **A경로 — 외부 제공 캡션 활용** (`caption_external.py`)
  AI 허브 민화 데이터셋처럼 라벨 JSON이 동봉된 경우 이를 파싱해 사용한다.
  **필드별 채택/제거 방침은 `docs/dataset-aihub-minhwa.md` 4장을 따른다.**
- **B경로 — 자동 캡셔닝** (`caption_auto.py`)
  라벨이 없는 임의 폴더용 기본 경로. WD14 tagger로 태그를 뽑은 뒤
  스타일 관련 태그를 규칙 기반으로 제거하고 트리거 워드를 선두에 삽입한다.

`caption.py`는 입력 폴더에 라벨 JSON이 있으면 A경로, 없으면 B경로를 자동 선택한다.
`--caption-mode {auto,external}`로 강제 지정할 수 있다.

> **[필수] 학습 캡션은 반드시 영문을 사용한다.**
> SD 1.5의 텍스트 인코더(CLIP ViT-L/14)는 영어 전용이다. 한글 캡션을 넣으면 학습이 실패한다.
> AI 허브 라벨의 `caption_kor`는 사람이 데이터를 검토할 때만 참고하고,
> 학습에는 `caption_eng`만 사용한다.

> **캡션의 원칙: 내용은 남기고 스타일은 지운다.**
> 캡션에 남은 것은 LoRA가 "변수"로 인식하고, 지워진 것은 "상수(=화풍)"로 흡수한다.
> 따라서 무엇이 그려졌는지(구도·객체)는 남기고, 어떻게 그렸는지(기법·재료·양식)는 제거한다.

### (3) 입력 데이터 품질 검증 및 서브셋 선별
폴더 안 이미지의 화풍이 섞여 있으면 LoRA가 무너진다.
`train` 실행 시 학습 전에 반드시 검증 단계를 거치고, 문제가 있으면 경고 후 사용자 확인을 받는다.
- 최소 장수 미달 (권장 15장 이상)
- 해상도 미달 (512px 미만)
- 중복/근접 중복 이미지 (perceptual hash)
- 종횡비 이상치

**서브셋 선별 (`subset.py`)**
대규모 데이터셋(수천 장)을 그대로 학습에 넣지 않는다. LoRA 학습에는 15~50장이면 충분하며,
오히려 하위 화풍이 섞이면 결과가 뭉개진다.

AI 허브 민화 데이터셋은 16개 화목으로 갈린다. **반드시 화목 하나만 선별해 학습한다.**

**선별은 디렉터리 선택으로 처리한다.** 데이터셋이 이미 화목별 · 기본/상세묘사별 ·
Training/Validation별로 폴더가 분리되어 있어, 파일명이나 캡션을 파싱할 필요가 없다.
(전체 경로 규칙은 `docs/dataset-aihub-minhwa.md` 2장)

선별 파이프라인:
1. `--filter HJ` → `Training\01.원천데이터\TS_기본데이터_09.화조도` 폴더로 해석
2. **상세묘사 폴더(`*_상세묘사데이터_*`)는 읽지 않는다** — 원본의 부분 확대본이므로
   중복 학습으로 스타일이 왜곡된다
3. 이미지 ↔ 라벨 페어링은 경로 치환(`01.원천데이터`→`02.라벨링데이터`, `TS_`→`TL_`).
   짝이 없는 항목은 경고 후 제외
4. 라벨 JSON의 구조화 필드로 2차 필터.
   **화조도 확정 조합: `drawing_type=채색` + `painting_type=일필+공필` → 411장**
   (필드명이 `painting_style`이 아니라 `painting_type`임에 주의 — 설명서 오타)
   복합값이 존재하므로 **완전 일치로만 매칭**한다. 부분 문자열 매칭 금지.
5. 품질 검증 통과분 중 `--limit` 장수만큼 샘플링

**순번에 결번이 있다.** 번호를 순회해 경로를 생성하지 말고 실제 디렉터리 목록을 읽는다.

선별 기준과 최종 선정 목록은 `data/prepared/{name}/subset_manifest.json`에 기록한다.

---

### (4) 평가에는 비교 기준선이 있다
AI 허브 데이터셋 구축기관이 동일 데이터로 수행한 레퍼런스 구현 결과가 공개되어 있다.

| 지표 | 레퍼런스 값 |
|---|---|
| CLIP Score | 0.74 |
| FID Score | 25.5 |

(모델: Stable Diffusion + LoRA Fine Tuning)

`evaluate/metrics.py`는 이 두 지표를 산출해 위 값과 대조한다.
**FID 계산의 실제 이미지 기준셋으로는 데이터셋의 Validation 세트
(`Validation\01.원천데이터\VS_기본데이터_{화목}`)를 사용한다.**
학습에 사용하지 않은 동일 화목 이미지이므로 평가 기준으로 적합하다.

학습 조건이 동일하지 않으므로 **우열을 주장하지 않는다.** "공개된 레퍼런스 대비 어느 위치인가"를
조건 차이와 함께 `docs/benchmarks.md`에 기록하는 것이 목적이다.

---

## 3. 핵심 제약 조건 (반드시 준수)

### VRAM 8GB (RTX 3060 Ti)
- 베이스 모델은 **SD 1.5 고정**. SDXL 학습 코드를 작성하지 않는다.
- 학습 기본값: `network_dim` 8~16, `batch_size` 1~2, `resolution` 512,
  `mixed_precision=fp16`, `gradient_checkpointing=true`, `cache_latents=true`
- **`enable_bucket=true` 필수.** 민화는 족자·병풍 형태가 많아 종횡비가 극단적이다
  (실측 예 317×1080, 약 1:3.4). 정사각 강제 리사이즈나 센터 크롭 시 구도가 파괴되어
  화풍 학습이 실패한다. `bucket_reso_steps=64`, `min_bucket_reso=256`, `max_bucket_reso=1024`.
  이에 따라 `scan.py`의 종횡비 이상치 기준도 1:4까지는 정상으로 간주한다.
- **학습과 ComfyUI는 동시에 실행할 수 없다.**
  `train` 실행 시 ComfyUI 프로세스를 자동 종료하고, 종료 후 필요 시 재기동한다.
  이 로직은 `train/runner.py`가 담당한다.

### Docker 미사용
- `Dockerfile`, `docker-compose.yml`을 만들지 않는다.
- Windows + 네이티브 실행 기준. 보조 스크립트가 필요하면 PowerShell로 작성한다.

### 가상환경 분리
- StyleForge 본체와 kohya_ss(sd-scripts)는 **별도 venv를 사용한다.**
  kohya는 서브프로세스로 호출하며, 의존성을 통합하지 않는다.
- ComfyUI 역시 프로젝트 외부에 설치되어 있고 이 저장소가 관리하지 않는다.

### 경로 하드코딩 금지
- ComfyUI 경로, kohya 경로, 데이터셋 루트, 모델·LoRA 출력 경로는 `.env` → `config.py`로 주입한다.
- 새 경로 추가 시 `.env.example`도 함께 갱신한다.
- 데이터셋 루트는 `DATASET_ROOT` 키를 사용한다. 실제 값과 주의사항은
  `docs/dataset-aihub-minhwa.md` 2장 참조.

---

## 4. 기술 스택

| 영역 | 스택 |
|---|---|
| 언어 | Python 3.11 |
| CLI | Typer |
| 학습 | kohya_ss (sd-scripts), SD 1.5 베이스 |
| 추론 | ComfyUI API 모드 (별도 프로세스) |
| 캡셔닝 | WD14 tagger (onnxruntime) |
| 평가 | CLIP (open_clip 또는 transformers), Pillow |
| 설정 | pydantic-settings, TOML |

**추론 백엔드로 diffusers가 아닌 ComfyUI를 쓰는 이유:**
ControlNet · LoRA · 업스케일 조합을 코드로 재구현하지 않고 워크플로우 JSON으로 교체 가능하게
유지하기 위함. 워크플로우는 코드가 아니라 데이터로 취급한다.

---

## 5. 디렉터리 구조

```
styleforge/
├── CLAUDE.md
├── README.md
├── .env.example
├── pyproject.toml
│
├── styleforge/
│   ├── cli.py                  # Typer 진입점 (train / apply / sweep)
│   ├── config.py               # .env 로드, 전역 설정
│   │
│   ├── dataset/
│   │   ├── scan.py             # 폴더 스캔 + 유효성 검사 (2-(3) 항목)
│   │   ├── subset.py           # 대규모 데이터셋 → 하위 장르 서브셋 선별
│   │   ├── preprocess.py       # 리사이즈 · 크롭 · 버킷팅
│   │   ├── caption.py          # 캡션 경로 자동 선택 (external / auto)
│   │   ├── caption_external.py # A경로: 외부 라벨 JSON 파싱 → 태그 변환
│   │   ├── caption_auto.py     # B경로: WD14 태깅
│   │   └── adapters/
│   │       └── aihub_minhwa.py # AI 허브 민화 데이터셋 라벨 스키마 어댑터
│   │
│   ├── train/
│   │   ├── config_builder.py   # kohya용 TOML 설정 생성
│   │   └── runner.py           # kohya 서브프로세스 실행 + VRAM 관리 + 로그 파싱
│   │
│   ├── apply/
│   │   ├── comfy_client.py     # ★ ComfyUI API 통신 (심장부)
│   │   ├── workflow_loader.py  # 워크플로우 JSON 템플릿 + 파라미터 주입
│   │   └── runner.py           # apply 실행 흐름
│   │
│   ├── sweep/
│   │   ├── planner.py          # 파라미터 조합 생성 (denoise × lora_weight × cn_strength)
│   │   └── grid.py             # 결과 비교 그리드 이미지 생성 (축 라벨 포함)
│   │
│   └── evaluate/
│       ├── metrics.py          # CLIP Score, FID, 원본 보존도
│       └── report.py           # 실행 결과 마크다운 리포트 (레퍼런스 기준선 대조 포함)
│
├── workflows/                  # ComfyUI API 포맷 JSON (수동 저장)
│   ├── style_transfer_lineart.json
│   └── mappings/
│       └── style_transfer_lineart.toml   # 노드 ID ↔ 파라미터 매핑 정의
│
├── configs/
│   └── train_default.toml      # 기본 학습 하이퍼파라미터
│
├── data/                       # gitignore
│   ├── refs/                   # 입력 레퍼런스 폴더
│   └── prepared/               # 전처리 결과 (캡션 포함)
│
├── outputs/                    # gitignore
│   ├── loras/                  # 학습된 LoRA (.safetensors)
│   ├── applied/                # 변환 결과 + meta.json
│   └── sweeps/                 # 비교 그리드 + 평가 리포트
│
└── docs/
    ├── user-guide.md           # 사용 매뉴얼
    ├── experiments.md          # 학습 실험 비교 기록 (최소 3회)
    └── benchmarks.md           # 8GB 제약 하 파라미터 선택 근거
```

---

## 6. ComfyUI 연동 규약

ComfyUI는 `--listen 127.0.0.1 --port 8188`로 기동하며, 주소는 `COMFY_URL` 환경변수로 주입한다.

### 사용 엔드포인트
- `POST /upload/image` — 입력 이미지 업로드
- `POST /prompt` — 워크플로우 제출, 응답으로 `prompt_id` 수신
- `GET /history/{prompt_id}` — 완료 결과 조회
- `GET /view?filename=...&subfolder=...&type=output` — 결과 이미지 다운로드
- `WS /ws?clientId={uuid}` — 진행률 수신

### WebSocket 메시지
- `type: "progress"` → `data.value` / `data.max`
- `type: "executing"` 이고 `data.node == null` → 해당 `prompt_id` 완료
- `type: "execution_error"` → 실패 처리, 사유를 결과 메타에 기록

### 워크플로우 JSON 취급 원칙 (중요)
- 워크플로우는 **코드가 아니라 데이터**다. ComfyUI에서 "Save (API Format)"으로 내보낸
  JSON을 `workflows/`에 그대로 둔다. 파이썬으로 재구현하지 않는다.
- 노드 ID를 코드에 하드코딩하지 않는다. `workflows/mappings/*.toml`에 매핑을 정의한다:

```toml
[params.positive_prompt]
node = "6"
field = "text"

[params.denoise]
node = "3"
field = "denoise"

[params.lora_weight]
node = "10"
field = "strength_model"
```

- `workflow_loader.py`는 JSON + 매핑을 읽어 지정된 노드의 inputs만 교체한다.
- 새 워크플로우 추가는 **JSON 1개 + 매핑 TOML 1개**로 끝나야 한다. 코드 수정 금지.

---

## 7. 명령별 동작 사양

### `train`
```
--input         레퍼런스 이미지 폴더 (필수)
--name          스타일 이름 = 트리거 워드 = LoRA 파일명 (필수)
--config        학습 설정 TOML (기본: configs/train_default.toml)
--filter        화목 코드 접두어 (예: HJ) — 대규모 데이터셋 사용 시
--meta-filter   라벨 필드 기준 2차 필터 (예: painting_style=공필,drawing_type=채색)
--include-detail  상세묘사 이미지도 포함 (기본: 제외)
--limit         학습에 사용할 최대 장수 (기본 40)
--caption-mode  auto | external (기본: 라벨 JSON 유무로 자동 판단)
--yes           검증 경고 무시하고 진행
```
흐름: 폴더 스캔·검증 → 서브셋 선별(`--filter` 지정 시) → 사용자 확인 →
전처리 → 캡셔닝 → kohya TOML 생성 →
ComfyUI 종료 → 학습 실행(진행률 표시) → LoRA를 `outputs/loras/`와 ComfyUI LoRA 폴더에 배치 →
학습 메타데이터(데이터 장수, 하이퍼파라미터, 소요 시간) 기록

### `apply`
```
--image     입력 이미지 (필수)
--style     LoRA 이름 (필수)
--strength  화풍 적용 강도 0.0~1.0 (기본 0.6) — 내부적으로 denoise/lora_weight로 매핑
--workflow  사용할 워크플로우 (기본: style_transfer_lineart)
--prompt    추가 프롬프트 (선택)
--seed      시드 (기본: 랜덤)
```
`--strength` 하나로 여러 내부 파라미터를 조절하는 것이 핵심이다.
사용자에게 3개 축을 노출하지 않는다. 매핑 곡선은 `apply/runner.py`에 정의한다.

결과는 `outputs/applied/{timestamp}_{style}/`에 이미지 + `meta.json`으로 저장한다.

### `sweep`
```
--image   입력 이미지 (필수)
--style   LoRA 이름 (필수)
--axes    탐색 축 (기본: denoise,lora_weight)
--steps   축당 분할 수 (기본 4)
```
흐름: 파라미터 조합 생성 → 순차 실행(동시 1개) → 비교 그리드 이미지 생성(축 라벨 포함) →
CLIP 기반 평가(스타일 유사도 / 원본 보존도) → 마크다운 리포트 →
권장 파라미터 조합 1개를 콘솔에 출력

**이 명령이 이 프로젝트의 차별점이다.** "감으로 튜닝"이 아니라
"자동 탐색 + 정량 평가 + 근거 있는 기본값 도출"임을 보여주는 기능이므로 대충 만들지 않는다.

---

## 8. 코딩 규약

- 타입 힌트 필수. `from __future__ import annotations` 사용.
- CLI 계층(`cli.py`)은 얇게 유지하고 로직은 각 모듈에 둔다.
- ComfyUI 통신은 반드시 `apply/comfy_client.py`를 경유한다. 다른 모듈에서 직접 HTTP 호출 금지.
- 장시간 작업(학습, sweep)은 진행 상황을 콘솔에 표시한다 (rich progress).
- 예외를 삼키지 않는다. 실패 시 원인을 로그와 메타데이터에 남긴다.
- 결과물을 저장할 때는 **항상 `meta.json`을 함께 쓴다** (입력 경로, 워크플로우, 전체 파라미터,
  시드, 소요 시간, LoRA 해시).
- 주석·커밋 메시지·CLI 출력 문구는 한국어. 로그 메시지는 영어.

---

## 9. 구현 순서 (한 번에 한 Phase씩)

- [x] **Phase 0** — `apply/comfy_client.py` 단독 검증.
      워크플로우 JSON 제출 → 진행률 수신 → 이미지 저장까지 스크립트로 확인.
      **이게 되기 전에 다른 파일을 만들지 않는다.**
- [x] **Phase 1** — `workflow_loader.py` + 매핑 TOML. `apply` 명령 동작 (기존 LoRA 사용).
- [x] **Phase 2** — `dataset/` 전체. 폴더 검증 → 서브셋 선별 → 전처리 → 캡셔닝(A·B 두 경로).
      AI 허브 라벨 승인 대기 중이라면 B경로(WD14)부터 구현하고 A경로는 나중에 붙인다.
- [x] **Phase 3** — `train/` 전체. kohya 서브프로세스 + VRAM 관리. `train` 명령 완성.
- [x] **Phase 4** — `sweep/` + `evaluate/`. 비교 그리드 + CLIP 평가 + 리포트.
- [ ] **Phase 5** — 실험 3회 이상 수행 후 `docs/experiments.md`, `benchmarks.md`, README 정리.

각 Phase 완료 후 실제 실행으로 검증하고 커밋한다. 다음 Phase 시작 전 컨텍스트를 정리한다.

---

## 10. 데이터 정책

### 일반 원칙
- 학습 데이터는 **공공 도메인 또는 이용 조건이 명시된 자료만** 사용한다.
- 생존 작가나 상용 IP의 화풍을 모사하는 데이터는 수집하지 않는다.
- 데이터 출처와 라이선스를 `docs/experiments.md`에 기록한다.
- `data/`, `outputs/`는 gitignore 대상이다. 저장소에 이미지를 커밋하지 않는다.

### 주 데이터셋: AI 허브 「한국 전통 민화 제작 데이터」 (2025 구축)
- 이미지 3,779set(기본) + 5,340set(상세묘사), 한글 캡션 28,872문장 / 영문 29,215문장
- 원본 소장처: 가회민화박물관, 국립민속박물관, 국립중앙박물관 등 50개 기관

> **파일명 규칙 · 라벨 JSON 스키마 · 화목 코드표 · 캡션 필드별 처리 방침은
> `docs/dataset-aihub-minhwa.md`에 정리되어 있다. 데이터 관련 구현 시 반드시 이 문서를 먼저 읽을 것.**

**이용 조건에 따른 준수 사항 (반드시 지킬 것)**
- 데이터 권리는 구축 수행기관 및 한국지능정보사회진흥원에 있으며, 본 프로젝트는
  **개인 포트폴리오 목적의 AI 학습 파이프라인 개발** 용도로만 사용한다.
- 판매 등 상업적 이용은 수행기관과 별도 협의가 필요하므로, 이 프로젝트 범위에서 시도하지 않는다.
- **원본 이미지·라벨 JSON을 저장소에 커밋하거나 재배포하지 않는다.**
- **학습된 `.safetensors` 가중치를 공개 저장소에 업로드하지 않는다.**
  (가중치 배포는 이용 조건상 회색지대이므로 회피한다)
- 공개하는 것은 **코드 · 생성 결과 이미지 · 문서**로 한정한다.
- README와 `docs/experiments.md`에 데이터 출처와 이용 조건 준수 내역을 명시한다.

### 대체 데이터 (승인 대기 중 사용)
AI 허브 신청 승인 전에는 국립중앙박물관 e뮤지엄 등 공공누리 조건이 명시된 자료로
Phase 0~2를 진행한다.

---

## 11. 하지 말아야 할 것

- Docker 관련 파일 생성
- SDXL 학습 코드 / 병렬 생성 처리
- 워크플로우 로직을 파이썬으로 재구현 (ComfyUI JSON을 그대로 쓸 것)
- 노드 ID·경로 하드코딩
- StyleForge와 kohya_ss의 의존성 통합
- 웹 UI, 서버, DB 추가 (이 프로젝트는 CLI 툴이다. 범위를 넓히지 않는다)
- 요청하지 않은 리팩터링, 파일 대량 선생성

---

## 12. 자주 쓰는 명령

```powershell
# 개발 환경
.\.venv\Scripts\activate

# 실행
styleforge train --input %DATASET_ROOT% --name hwajodo --filter HJ --meta-filter drawing_type=채색,painting_type=일필+공필 --limit 40
styleforge apply --image .\samples\photo.jpg --style hwajodo --strength 0.6
styleforge sweep --image .\samples\photo.jpg --style hwajodo

# ComfyUI 수동 기동 (프로젝트 외부 경로)
# python main.py --listen 127.0.0.1 --port 8188

# 테스트
pytest
```
