# Versioned ML artifacts

검수·재현에 필요한 데이터셋과 정책 체크포인트만 이 디렉터리에 복사해 Git LFS로 관리합니다.
실행 중 계속 덮어쓰는 `outputs/`, 캐시, 임시 프레임은 기존처럼 Git에서 제외합니다.

권장 구조:

```text
artifacts/
├─ datasets/
│  └─ so101_sim_pick_v22/
└─ checkpoints/
   ├─ act_pick_v21/
   │  └─ 015000/pretrained_model/
   └─ act_pick_v22/
      └─ 015000/pretrained_model/
```

`*.safetensors`, `*.mp4`, `*.parquet` 등 대용량 파일은 `.gitattributes` 규칙에 따라
Git LFS에 저장됩니다. JSON 설정과 메타데이터는 일반 Git으로 관리합니다.

원래 Mac에서 우선 전달할 항목:

1. `outputs/train/act_pick_v22/checkpoints/015000/pretrained_model/`
2. `outputs/train/act_pick_v21/checkpoints/015000/pretrained_model/`
3. v2.2 학습에 사용한 200에피소드 데이터셋
4. 필요 시 v1·v2 체크포인트와 비교 평가 자료

API 키, 토큰, `.env`, Hugging Face 인증 파일은 절대 이 디렉터리에 넣지 않습니다.
