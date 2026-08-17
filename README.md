# StyleForge
[요약](https://claude.ai/code/artifact/bf8c3891-9f48-41e3-b1a1-f1509dc79308)



화풍 학습 · 적용 자동화 CLI 툴 — 레퍼런스 이미지 폴더 하나로 화풍 LoRA를 학습하고,
임의의 입력 이미지를 그 화풍으로 변환합니다. 학습 → 적용 → 파라미터 탐색까지
한 파이프라인으로 연결하는 게 목표입니다.

```bash
# 1) 폴더 → LoRA 학습
styleforge train --input ./refs/minhwa --name minhwa

# 2) 이미지 → 화풍 변환
styleforge apply --image photo.jpg --style minhwa --strength 0.6

# 3) 파라미터 탐색 + 비교 그리드 생성
styleforge sweep --image photo.jpg --style minhwa
```

사용자는 `denoise`/`rank`/`lr` 같은 내부 파라미터를 몰라도 됩니다. 모든 실행은
결과물 옆에 `meta.json`을 남겨 재현 가능하고, `sweep`이 파라미터 조합을 자동
탐색해 감이 아니라 수치로 튜닝 근거를 남깁니다. 설계 배경은 [CLAUDE.md](CLAUDE.md)에
자세히 정리되어 있습니다.

**성능:** `apply`/`sweep`은 입력 이미지를 원본 비율을 유지한 채 SD1.5 기준
(512px)으로 자동 리사이즈합니다. 폰카메라 원본(4032×3024)처럼 큰 이미지를
리사이즈 없이 그대로 diffusion에 넣던 과거 버전은 VAE가 8GB VRAM에서 OOM →
타일 모드로 폴백해 극도로 느려졌는데(스텝당 43초), 이 최적화로 동일 조건에서
apply 1회 약 **136배**, sweep 1회(16조합) 약 **250배 이상** 단축했습니다
(실측 수치: [`docs/benchmarks.md`](docs/benchmarks.md) 3장).

---

## 요구 사항

- Windows + NVIDIA GPU (개발 기준: RTX 3060 Ti, VRAM 8GB)
- Python 3.11+
- 별도로 설치되어 있어야 하는 것 (이 저장소가 관리하지 않음)
  - [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — `--listen 127.0.0.1 --port 8188`로 기동
  - [kohya_ss(sd-scripts)](https://github.com/kohya-ss/sd-scripts) — **StyleForge와 별도 venv**에 설치
  - SD 1.5 체크포인트 (`v1-5-pruned-emaonly.safetensors` 등)
  - ControlNet lineart 모델 (`control_v11p_sd15_lineart.pth`) + `comfyui_controlnet_aux` 커스텀 노드

Docker는 쓰지 않습니다. SDXL은 지원하지 않습니다 (SD 1.5 고정, VRAM 8GB 제약).

---

## 설치

### 1) StyleForge 본체

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

CLIP 기반 평가(`sweep`의 스타일 적용도 계산)까지 쓰려면 추가로:

```powershell
# torch는 PyPI 기본 인덱스가 CUDA 포함 빌드를 주므로 CPU 전용 인덱스로 받는다.
# evaluate는 학습/추론과 VRAM을 다투지 않아도 되는 작업이라 CPU로 충분하다.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch pytorch-fid
```

### 2) kohya_ss(sd-scripts) — 별도 venv

StyleForge 본체와 의존성을 통합하지 않습니다. 학습은 서브프로세스로 호출합니다.

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git D:\kohya_ss
cd D:\kohya_ss
python -m venv venv
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

> `requirements.txt`의 `safetensors==0.4.5` 핀이 최신 Python에서는 소스 빌드를
> 시도하다 실패할 수 있다. 그 경우 `safetensors>=0.4.5`로만 완화하고 나머지
> 핀(특히 `transformers`/`diffusers`/`accelerate`)은 그대로 둘 것 — 메이저
> 버전을 올리면 체크포인트 로딩 시 state_dict 키가 어긋나며 깨진다.

### 3) `.env` 설정

`.env.example`을 복사해 `.env`로 만들고 실제 경로를 채웁니다.

```powershell
copy .env.example .env
```

| 키 | 설명 |
|---|---|
| `COMFY_URL` | ComfyUI API 주소 |
| `DATASET_ROOT` | AI 허브 민화 데이터셋 루트 (대규모 데이터셋 사용 시) |
| `KOHYA_PYTHON` / `KOHYA_SCRIPT_DIR` | 위에서 만든 kohya venv의 `python.exe`와 저장소 경로 |
| `SD15_CHECKPOINT_PATH` | kohya 학습용 SD1.5 체크포인트 |
| `COMFY_START_COMMAND` / `COMFY_START_CWD` | `train` 실행 시 ComfyUI 자동 종료 후 재기동에 쓰는 명령 |
| `COMFY_LORA_DIR` | 학습된 LoRA를 복사해 넣을 ComfyUI의 `models/loras` |

나머지 키 설명은 `.env.example` 주석 참고.

---

## 사용법

### `train` — 폴더 → LoRA

```powershell
# 임의 폴더 (이미지 옆에 같은 이름의 .json 라벨이 있으면 자동으로 그 라벨을 캡션 소스로 씀)
styleforge train --input .\refs\minhwa --name minhwa

# AI 허브 대규모 데이터셋에서 화목 하나만 선별해 학습
styleforge train --input %DATASET_ROOT% --name hwajodo --filter HJ `
  --meta-filter drawing_type=채색,painting_type=일필+공필 --limit 40
```

흐름: 폴더 스캔·검증 → (─`filter` 지정 시) 서브셋 선별 → 캡셔닝 → 전처리 →
kohya 설정 생성 → ComfyUI 자동 종료 → kohya 학습 → LoRA를 `outputs/loras/`와
ComfyUI `models/loras`에 배치 → ComfyUI 재기동.

<img width="886" height="306" alt="image" src="https://github.com/user-attachments/assets/e2121cb9-93e7-485d-ab61-cc0537e5668b" />
해당 데이터셋을 사용했습니다.

### `apply` — 이미지 → 화풍 변환

```powershell
styleforge apply --image .\samples\photo.jpg --style hwajodo --strength 0.6
```

`--strength`(0.0~1.0) 하나로 `denoise`/`lora_weight`/`controlnet_strength`
세 축을 함께 조절합니다. 결과는 `outputs/applied/{timestamp}_{style}/`에
이미지와 `meta.json`으로 저장됩니다.

입력 이미지는 원본 해상도 그대로 diffusion에 들어가지 않고, 원본 비율을
유지한 채 긴 변을 SD1.5 기준(512px)에 맞춰 자동 리사이즈됩니다. 폰카메라
사진처럼 큰 원본을 그대로 넣어도 느려지지 않습니다 — 리사이즈가 없던
과거 버전 대비 실측 성능 차이는 [`docs/benchmarks.md`](docs/benchmarks.md)
3장 참고.

<img width="453" height="677" alt="images" src="https://github.com/user-attachments/assets/eeffec35-50e2-4f46-a7ef-b7338b9e0ef7" />
원본 이미지
<img width="344" height="512" alt="styleforge_apply_00076_" src="https://github.com/user-attachments/assets/fbb7bab4-e1cf-485f-ae27-e32ae089018d" />
화풍 변환 후

### `sweep` — 파라미터 탐색 + 비교 그리드

```powershell
styleforge sweep --image .\samples\photo.jpg --style hwajodo
```

`denoise`×`lora_weight` 격자(기본 4×4)를 순차 실행하고, CLIP 기반으로
스타일 적용도/원본 보존도를 채점해 축 라벨 붙은 비교 그리드 이미지와
마크다운 리포트(`report.md`)를 `outputs/sweeps/{timestamp}_{style}/`에
남깁니다. 권장 조합 1개를 콘솔에 출력합니다.

<img width="1186" height="1156" alt="grid" src="https://github.com/user-attachments/assets/3b5ef5f5-e1c1-4914-bf21-c810f7449e19" />

### GUI로 쓰기

명령줄 대신 데스크톱 GUI로도 같은 3개 기능(학습/적용/탐색)을 쓸 수 있습니다.

```powershell
styleforge-gui
```

탭으로 나뉜 창이 뜨고, 각 탭이 위 `train`/`apply`/`sweep` 명령과 1:1로
대응합니다. 새 서버나 브라우저를 띄우지 않는 로컬 전용 창입니다.

상단 바에는 ComfyUI 실행 상태(켜짐/꺼짐)가 표시되고, 꺼져 있으면 **켜기**
버튼으로 바로 기동할 수 있습니다(`COMFY_START_COMMAND` 필요). **설정**
버튼을 누르면 ComfyUI 실행 파일 폴더·LoRA 폴더 경로를 찾아보기로 골라
`.env`에 바로 저장할 수 있습니다 — 외장 드라이브 연결이 바뀌는 등으로
경로가 자주 깨지는 문제를 텍스트 에디터 없이 고칠 수 있게 하기 위함입니다.

#### 배포용 실행 파일 만들기

터미널 없이 더블클릭만으로 GUI를 켜고 싶다면 `StyleForge.exe`를 빌드할 수
있습니다. **이 exe는 `.venv`를 다시 담지 않는 얇은 런처**입니다 — 위
[설치](#설치) 과정을 이미 끝낸(=`.venv`가 준비된) 컴퓨터에서만 동작하며,
그 컴퓨터의 `.venv\Scripts\pythonw.exe`로 GUI를 대신 실행해줄 뿐입니다.
그래서 torch 등 무거운 의존성을 다시 묶지 않아 빌드가 몇 초 안에 끝나고
결과물도 10MB 안팎으로 작습니다. (설치 없이 어디서나 도는 완전한 단일
배포판이 필요하다면 이 방식으로는 부족합니다 — evaluate 의존성(torch)까지
통째로 묶는 PyInstaller 빌드가 별도로 필요하며, 결과물이 수 GB로 커지고
onnxruntime/torch 조합이 빌드 환경에 따라 깨지기 쉽습니다.)

```powershell
pip install -e ".[build]"
pyinstaller --onefile --windowed --name StyleForge --distpath . --workpath build --specpath build scripts/launch_gui.py
```

`StyleForge.exe`가 프로젝트 루트(`.venv`, `.env`가 있는 위치)에 생성됩니다.
**이 exe는 반드시 그 자리에 두고 실행해야 합니다** — 자기 위치를 기준으로
`.venv`를 찾기 때문입니다. `.venv`가 없는 채로 실행하면 안내 메시지 창이 뜹니다.

---
<img width="812" height="664" alt="image" src="https://github.com/user-attachments/assets/87dbe5ba-2177-4d8d-bde6-9d2ea20970a3" />

<img width="806" height="657" alt="image" src="https://github.com/user-attachments/assets/4d550460-7f05-4dba-a25c-d6bc2af46a9d" />

<img width="811" height="665" alt="image" src="https://github.com/user-attachments/assets/84509268-b913-4b77-be9a-4d8fcd0665a8" />

## 문서

- [`CLAUDE.md`](CLAUDE.md) — 프로젝트 설계 철학, 기술적 결정 배경, 구현 순서
- [`docs/dataset-aihub-minhwa.md`](docs/dataset-aihub-minhwa.md) — AI 허브
  민화 데이터셋 실측 스펙(공식 설명서와의 불일치 포함), 화목 코드표
- [`docs/experiments.md`](docs/experiments.md) — 학습·튜닝 실험 기록
- [`docs/benchmarks.md`](docs/benchmarks.md) — 8GB VRAM 제약 하 파라미터
  선택 근거, apply/sweep 성능 실측(이미지 리사이즈 수정 전/후 비교),
  AI 허브 공개 레퍼런스 대비 위치
---
## ComfyUI 워크플로우 캡쳐
<img width="1718" height="565" alt="image" src="https://github.com/user-attachments/assets/ba58a50a-9664-4b39-9848-c07dbaf8a9b9" />

입력 사진 불러오기
   ↓
윤곽선(lineart) 추출
   ↓
ControlNet에 윤곽선 넘기기
   ↓
화풍 LoRA 적용
   ↓
디노이즈 강도만큼 다시 그리기
   ↓
결과 이미지 저장
---

## 데이터·모델 이용 정책

- 학습 데이터는 공공 도메인 또는 이용 조건이 명시된 자료만 사용합니다.
  주 데이터셋인 AI 허브 「한국 전통 민화 제작 데이터」의 이용 조건과 준수
  내역은 `docs/experiments.md` 부록에 명시되어 있습니다.
- **원본 이미지·라벨 JSON, 학습된 `.safetensors` 가중치는 이 저장소에
  커밋하거나 공개 배포하지 않습니다.** `data/`, `outputs/`는 `.gitignore`
  대상입니다.
- 공개하는 것은 코드·생성 결과 이미지(예시)·문서로 한정합니다.
- 생존 작가나 상용 IP의 화풍을 모사하는 데이터는 수집하지 않습니다.

이 프로젝트는 개인 포트폴리오 목적의 AI 학습 파이프라인 개발 용도로만
사용됩니다.
