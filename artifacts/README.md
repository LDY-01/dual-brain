# 재현용 ML 산출물

현재 기준선 평가와 6cm·DAgger 실험 재현에 필요한 최소 산출물을 Git LFS로 관리합니다.
실행 중 계속 덮어쓰는 `outputs/`, `results/`, 캐시와 임시 프레임은 Git에서 제외합니다.

현재 구조:

```text
artifacts/
├─ datasets/
│  ├─ so101_sim_pick_v22/
│  ├─ so101_sim_pick_v24/
│  └─ so101_sim_pick_dagger_v2_balanced50/
└─ checkpoints/
   └─ act_pick_v22/
      └─ pretrained_model/
```

`*.safetensors`, `*.mp4`, `*.parquet` 등 대용량 파일은 `.gitattributes` 규칙에 따라
Git LFS에 저장됩니다. JSON 설정과 메타데이터는 일반 Git으로 관리합니다.

포함 기준:

| 항목 | 에피소드·프레임 | 크기 | 용도 |
|---|---:|---:|---|
| `act_pick_v22/pretrained_model` | 모델·설정 7개 파일 | 약 197.15MB | 현행 정책 후보의 추론·평가·파인튜닝. 48%는 2026-08-12 코드의 과거 기준선이며 현재 코드 재평가는 32% |
| `so101_sim_pick_v22` | 200ep·38,600프레임 | 약 139.13MB | Mac v2.2 원본 손목 카메라 데이터 |
| `so101_sim_pick_v24` | 200ep·48,800프레임 | 약 169.47MB | 개선된 6cm 교사 성공 데이터 |
| `so101_sim_pick_dagger_v2_balanced50` | 50ep·25,924프레임 | 약 93.75MB | 집기·운반·놓기 실패 유형별 균형 교정 |

전체 재현 자산은 26개 파일, 약 599.5MB입니다.

사용 방법:

```powershell
git lfs install
git lfs pull
```

v2.2 기준선 평가 예시:

```powershell
uv run python kwon_lab/eval/eval_act_pick.py --version v2.2 --step act_pick_v22 --checkpoint-root artifacts/checkpoints --n-action-steps 10 --episodes 20
```

옵티마이저를 포함한 전체 12GB 학습 체크포인트, v1·v2.1 구버전 모델, 평가 영상 모음,
폐기·중단된 데이터셋과 별도 종합보고서는 포함하지 않습니다.

API 키, 토큰, `.env`, Hugging Face 인증 파일은 절대 이 디렉터리에 넣지 않습니다.
