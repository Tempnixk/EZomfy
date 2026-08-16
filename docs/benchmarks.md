# 벤치마크 — 8GB VRAM 제약 하 파라미터 근거 및 레퍼런스 대비 위치

> 이 문서의 목적은 "레퍼런스보다 낫다/못하다"를 주장하는 게 아니다. 학습 조건이
> 동일하지 않으므로 우열은 판단하지 않고, **공개된 레퍼런스 대비 우리 설정이
> 대략 어느 위치에 있는지**를 조건 차이와 함께 기록한다 (CLAUDE.md 2-(4)).

---

## 1. 8GB VRAM(RTX 3060 Ti) 제약 하 학습 파라미터 근거

`configs/train_default.toml`의 기본값과 그 이유:

| 파라미터 | 값 | 근거 |
|---|---|---|
| base model | SD 1.5 | SDXL은 8GB에서 LoRA 학습이 사실상 불가능(base UNet만 여러 GB) — CLAUDE.md 3장에서 SDXL 코드 자체를 금지 |
| `resolution` | 512 | SD1.5 네이티브 해상도. 이 이상 올리면 latent 캐시·activation 메모리가 제곱으로 늘어 8GB에서 batch_size=1도 위태로워짐 |
| `network_dim` | 8 | LoRA rank. 8~16 권장 구간의 하한 — 8GB에서 여유를 더 남기려는 선택. rank가 학습 품질에 미치는 영향은 E2(`docs/experiments.md`, 예정)에서 추가 검증 예정 |
| `network_alpha` | 4 | `network_dim`의 절반. kohya 권장 관례(alpha=dim/2)를 따름 — dim과 별도로 조정할 근거가 아직 없어 관례값 유지 |
| `train_batch_size` | 1 | 8GB에서 batch 2 이상은 `resolution=512` + `network_dim=8` 조합에서도 OOM 위험이 커 가장 보수적인 값을 기본으로 함 |
| `mixed_precision` | fp16 | fp32 대비 activation 메모리를 절반으로 줄임 — 8GB에서는 사실상 필수 |
| `gradient_checkpointing` | true | activation을 저장하지 않고 backward 시 재계산 — VRAM을 크게 아끼는 대신 학습 속도가 떨어지는 트레이드오프. 8GB에서는 VRAM이 항상 더 급한 자원이라 켜둠 |
| `cache_latents` | true | 매 스텝 VAE 인코딩을 반복하지 않고 미리 계산해 캐싱 — VRAM보다는 학습 속도 최적화지만, VAE를 학습 루프 중 계속 GPU에 올려둘 필요가 없어져 간접적으로 VRAM에도 도움 |
| `optimizer_type` | AdamW8bit | bitsandbytes의 8bit 옵티마이저. 일반 AdamW 대비 옵티마이저 상태(모멘텀·분산) 메모리를 1/4로 줄임 — 8GB에서 다른 여유를 만들어주는 항목 |
| `enable_bucket` | true | 민화 데이터셋은 족자·병풍 형태로 종횡비가 극단적(실측 1:3.4)이라 정사각 강제 리사이즈 시 구도가 파괴됨 (`docs/dataset-aihub-minhwa.md` 3-2장). VRAM보다는 데이터 품질 문제지만, 여기 함께 기록 |

### 실측: 실제 학습 실행 (hwajodo, 2026-08-13)

| 항목 | 값 |
|---|---|
| 데이터 | 화조도(HJ), `drawing_type=채색`+`painting_type=일필+공필`, 40장 |
| 총 스텝 | 4,000 (40장 × 10 repeats × 10 epoch, batch_size=1) |
| 소요 시간 | 5시간 10분 (18,618.88초) |
| 스텝당 속도 | 초반 ~11s/it → 후반 ~4.6s/it (가속 — 원인 미분석, 디스크 캐시·버킷 재사용 등으로 추정) |
| 결과 | OOM 없이 10 epoch 완료, `outputs/loras/hwajodo.safetensors` (18.9MB) 생성 |
| GPU | RTX 3060 Ti, VRAM 8192MB, 드라이버 CUDA 13.1, torch 2.13.0+cu130 |

**피크 VRAM 수치는 별도로 로깅하지 않았다** (kohya 로그에 VRAM 사용량 출력이
없었음) — 다만 위 설정으로 OOM 없이 완주했다는 사실 자체가 "이 조합이 8GB
안에서 동작한다"는 실증이다. 정확한 피크 값이 필요하면 `nvidia-smi
--query-gpu=memory.used --format=csv -l 1`을 학습과 병행 실행해 로깅하는
것을 후속 과제로 남긴다.

---

## 2. apply 파라미터 매핑 곡선 근거

`apply/runner.py`의 `strength_to_params()`는 `--strength`(0~1)를 아래 구간에
선형 매핑한다:

| 파라미터 | strength=0 | strength=1 |
|---|---|---|
| `denoise` | 0.35 | 0.80 |
| `lora_weight` | 0.40 | 1.00 |
| `controlnet_strength` | 0.60 | 0.95 |

**E4 실험(`docs/experiments.md`)으로 이 곡선의 방향성을 실측 검증했다:**
- `denoise`가 style/preservation 트레이드오프의 지배적 변수임을 확인 — 곡선에서
  denoise를 strength에 선형 비례시키는 현재 설계가 타당함을 재확인했다.
- `lora_weight`는 이번 표본(풍경/인물/정물 각 1장)에서 뚜렷한 방향성을
  못 찾았다 — 곡선을 바꿀 근거가 부족해 현재 값을 유지한다. 표본을 늘린
  재검증을 후속 과제로 남긴다.
- `controlnet_strength`의 on/off 대조군 비교(0.0 vs 0.6)는 `sweep`이 그리드를
  2차원으로 유지하기 위해 축을 최대 2개로 제한하는 구조라 이번에는 다루지
  못했다. 후속 과제.

---

## 3. AI 허브 공개 레퍼런스 대비 CLIP Score / FID

구축기관이 동일 데이터로 수행한 레퍼런스 구현(Stable Diffusion + LoRA Fine
Tuning)의 공개 수치:

| 지표 | 레퍼런스 |
|---|---|
| CLIP Score | 0.74 |
| FID | 25.5 |

### 우리 측정값

`evaluate/metrics.py`의 `clip_text_score()`(Hessel et al. CLIPScore 정의,
w=2.5)와 `compute_fid()`(pytorch-fid, InceptionV3 pool3 활성값)로 산출했다.
생성 이미지는 새로 만들지 않고 **E4 스윕에서 이미 생성된 48장**(풍경/인물/
정물 각 16조합, `hwajodo` LoRA, 프롬프트 `"hwajodo, best quality"`)을
재사용했고, 실사 기준셋은 `Validation\01.원천데이터\VS_기본데이터_09.화조도`
91장을 그대로 썼다.

| 지표 | 우리 측정값 | 레퍼런스 | 표본 수 |
|---|---:|---:|---|
| CLIP Score | 0.5609 (min 0.5106, max 0.6080) | 0.74 | 생성 48장 |
| FID | 386.21 | 25.5 | 생성 48장 vs 실사 91장 |

FID 386.21은 레퍼런스(25.5)와 자릿수부터 다른데, 아래 캐비앗 중 특히
**도메인 불일치**가 지배적 원인으로 보인다 — 48장 중 32장(인물·정물)이
화조도와 무관한 입력(게임 캐릭터 렌더, 상품 사진)에서 나온 img2img
결과라, 화조도만 담긴 91장 실사 세트와 분포 자체가 크게 벌어진다.
CLIP Score(0.5609)도 같은 이유로 레퍼런스보다 낮게 나왔을 가능성이 크다.

### 이 수치를 읽을 때 반드시 감안할 것 (우열 판단 근거로 쓰지 않는 이유)

- **표본 수가 논문 수준 FID(보통 수천~수만 장)에 비해 훨씬 작다.** 48장·91장
  규모의 FID는 분산이 매우 커서 절대값 자체의 신뢰구간이 넓다.
- **생성 조건이 레퍼런스와 다르다.** 레퍼런스는 프롬프트 기반 텍스트-이미지
  생성으로 추정되는 반면, 우리 측 48장은 `sweep`이 만든 **img2img** 결과이고
  그중 32장(인물·정물)은 화조도와 콘텐츠 도메인이 먼 입력(게임 캐릭터 렌더,
  상품 사진)에서 나왔다 — E4에서 이미 "도메인이 먼 입력에서는 CLIP 스타일
  지표 신뢰도가 낮다"고 확인한 바로 그 데이터다.
- **학습 규모도 다르다.** 우리 LoRA는 40장·10epoch(rank 8)로 학습했고,
  레퍼런스의 정확한 학습 규모는 공개 자료에 명시되어 있지 않다.

**결론: 이 수치는 "같은 스크립트로 잰 참고용 좌표"일 뿐, 레퍼런스 대비
우열 판단에는 쓰지 않는다.** 공정한 비교를 하려면 (1) 순수 txt2img로
레퍼런스와 동일한 방식으로 생성하고, (2) 화조도 도메인 프롬프트로만 표본을
구성하고, (3) 표본 수를 최소 수백 장으로 늘리는 재실험이 필요하다 — 후속
과제로 남긴다.

---

## 4. 후속 과제 요약

1. `network_dim`/learning rate 조합별 VRAM·품질 비교 (E2, 예정)
2. `lora_weight` 축 재검증 — 더 많은 표본으로 방향성 확인
3. `controlnet_strength` on/off 대조군 비교 — `sweep` 2축 제한을 벗어난 별도 실험
4. 학습 중 피크 VRAM 실측 로깅
5. 순수 txt2img 기반 CLIP Score/FID 재측정 (레퍼런스와 조건 최대한 맞춰서)
