# SO101 렌즈 뚜껑 넣기 프로젝트

SO101 리더-팔로워 로봇을 사용해 **렌즈 뚜껑을 열린 상자 안에 넣는 작업**을 시연 데이터로 학습시키고, 학습된 ACT 정책으로 팔로워암을 자율 실행하는 프로젝트입니다.

> 이 문서는 2026-07-31 기준의 작업 기록과 재현 절차입니다. 실제 로봇을 움직이기 전에 항상 주변을 비우고, 비정상 진동이나 충돌이 발생하면 팔로워의 12V 전원을 즉시 분리합니다.

## 현재 상태

- SO101 리더암·팔로워암 조립, USB 포트 식별, 캘리브레이션 완료
- Innomaker U20CAM-720P 카메라 연결 및 학습 장면 고정 완료
- 데이터셋 `DY-01/lens_cap_into_box_v3` 수집·검수·Hugging Face 업로드 완료
- ACT 모델 `act_lens_cap_v1` 학습 완료
- 팔로워암 단독 자율 실행(rollout) 확인 완료
- 실행 환경을 Windows에서 macOS로 이전
  - follower 포트: `/dev/tty.usbmodem5B140307781`
  - 전방 카메라: OpenCV index `0`
  - Windows에서 사용한 follower 캘리브레이션 파일 이전 완료
- 평가용 rollout 데이터셋을 수집 중
  - Windows 기록: 정상 완료 에피소드 9개, 중간 종료된 부분 에피소드 1개
  - macOS 예비 테스트에서는 성공률이 약 10%로 관찰됐으나 정식 평가 전이므로 확정값이 아님
  - 뚜껑 위치가 학습 위치에서 조금만 벗어나도 집기에 실패하고, 그리퍼에 뚜껑을 보정해 주면 상자에 넣는 후반 동작은 수행함

현재 모델은 작업 방향을 일부 재현하지만, 로컬 CPU 추론이 느리고 관절 명령이 안전 제한에 자주 걸립니다. 따라서 현재 단계의 목적은 **모델의 실제 성공률을 측정하고 실패 원인을 분류하는 것**입니다.

## 하드웨어 구성

| 구성품 | 역할 | 현재 macOS 설정 |
|---|---|---|
| SO101 follower | 자율 실행 팔 | `/dev/tty.usbmodem5B140307781`, 별도 12V 전원 |
| SO101 leader | 시연 데이터 수집용 팔 | 아직 macOS 포트 미확인 (자율 실행에는 불필요) |
| Feetech STS3215-12V | 팔로워 관절 서보 6개 | ID 1~6 |
| Innomaker U20CAM-720P | 전방 카메라 | OpenCV index `0`, 640x480, 30 FPS |

이전 Windows 설정은 follower `COM3`, leader `COM5`, 전방 카메라 index `1`이었습니다.

팔로워 관절 ID는 다음과 같습니다.

| ID | 관절 |
|---:|---|
| 1 | `shoulder_pan` |
| 2 | `shoulder_lift` |
| 3 | `elbow_flex` |
| 4 | `wrist_flex` |
| 5 | `wrist_roll` |
| 6 | `gripper` |

### 연결 원칙

- 포트 이름과 OpenCV 카메라 index는 OS와 물리적 연결 순서에 따라 바뀔 수 있으므로, 연결 구성을 바꾼 뒤에는 반드시 다시 확인합니다.
- 현재 MacBook에는 USB-A 포트가 없어 카메라를 USB-C 허브에 연결합니다. 프레임 시간초과가 발생하면 다른 장치를 분리하거나, 로봇과 카메라를 서로 다른 USB-C 포트/허브로 나누거나, 전원형 허브를 사용합니다.
- 자율 실행 시에는 팔로워와 카메라만 연결합니다. 리더암은 필요하지 않습니다.
- 관절을 손으로 만지거나 케이블을 만질 때는 팔로워 12V 전원을 먼저 분리합니다.

## 개발 환경

- 현재 실행 OS: macOS (Apple Silicon)
- 이전 실행 OS: Windows
- Python: 3.12
- 패키지 관리: `uv`
- LeRobot: `0.6.0`
- 로컬 자율 실행 장치: CPU
- 학습 환경: Google Colab Tesla T4 GPU

프로젝트 의존성은 다음 파일에서 관리합니다.

- `pyproject.toml`
- `uv.lock`
- `.python-version`

프로젝트 명령은 항상 이 폴더에서 실행합니다.

```bash
cd /Users/doyounglim/Desktop/SO101
```

## 캘리브레이션

리더와 팔로워는 각각 별도로 캘리브레이션합니다. 기존 캘리브레이션이 있으면, 재실행 시 표시되는 질문에서 `Enter`를 눌러 기존 파일을 사용합니다. `c`를 입력하면 새 캘리브레이션이 시작됩니다.

현재 follower는 Windows에서 실제 학습·실행에 사용한 캘리브레이션 파일을 다음 macOS 경로로 이전해 사용합니다.

```text
/Users/doyounglim/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json
```

실행 중 기존 캘리브레이션을 사용할지 묻는 경우 `Enter`를 누릅니다. `c`를 누르면 현재 파일을 새 캘리브레이션으로 덮어쓰므로, 모터 교체나 조립 변경처럼 재캘리브레이션이 필요한 경우가 아니면 누르지 않습니다.

팔로워의 3번 서보(`elbow_flex`)는 내부 기어 파손으로 교체했으며, 교체 후 ID 3 설정과 팔로워 재캘리브레이션을 완료했습니다.

## 카메라·포트 확인

```bash
# 카메라 index 확인
uv run lerobot-find-cameras opencv

# 특정 모터 버스의 포트 찾기
uv run lerobot-find-port
```

현재 macOS 작업 카메라는 `Camera #0`입니다. 카메라를 다른 USB 포트로 옮겼다면 결과를 확인한 뒤 모든 명령의 `index_or_path` 값을 바꿔야 합니다.

## 시연 데이터셋

### 작업 정의

```text
Place the lens cap into the open box.
```

### 최종 학습 데이터

| 항목 | 값 |
|---|---|
| Hugging Face dataset | `DY-01/lens_cap_into_box_v3` |
| 에피소드 수 | 50 |
| 총 프레임 | 45,000 |
| 에피소드 길이 | 30초 |
| 목표 FPS | 30 |
| 카메라 | 전방 카메라 1대, 640x480 |

데이터셋에는 영상뿐 아니라 다음 정보가 같이 저장됩니다.

- `observation.images.front`: 카메라 영상
- `observation.state`: 팔로워의 실제 6개 관절 상태
- `action`: 리더암 시연에서 나온 6개 관절 명령
- `timestamp`, `frame_index`, `episode_index`, `task_index`

따라서 ACT는 영상만으로 학습하는 것이 아니라, **카메라 영상 + 현재 관절 상태를 보고 다음 관절 동작을 예측**합니다.

### 시연 수집 예시

```powershell
uv run lerobot-record --robot.type=so101_follower --robot.port=COM3 --robot.id=so101_follower --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=so101_leader --dataset.repo_id=DY-01/<dataset_name> --dataset.single_task="Place the lens cap into the open box." --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10 --dataset.push_to_hub=false --display_data=true
```

시연 영상은 성공한 동작만 보관합니다. 손이 화면에 오래 남거나, 집기·놓기가 실패했거나, 기계적 이상이 발생한 에피소드는 학습 데이터에서 제외합니다.

## ACT 학습

ACT(Action Chunking Transformer)는 카메라 영상과 현재 관절 상태를 입력으로 받고, 이후 관절 명령 묶음(action chunk)을 예측하는 정책입니다.

Google Colab T4 GPU에서 학습했습니다.

```bash
pip install -q "lerobot[training]==0.6.0"

lerobot-train \
  --dataset.repo_id=DY-01/lens_cap_into_box_v3 \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/content/drive/MyDrive/SO101/act_lens_cap_v1 \
  --job_name=act_lens_cap_v1 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=5000 \
  --save_freq=1000 \
  --log_freq=50
```

초기 5,000 step 학습 후 동일 모델을 재개하여 총 20,000 step까지 학습했습니다.

| 항목 | 값 |
|---|---|
| 정책 | ACT |
| 총 학습 step | 20,000 |
| 최종 training loss | 약 0.16 |
| 모델 이름 | `act_lens_cap_v1` |

학습 loss는 시연 데이터를 얼마나 잘 재현하는지의 지표일 뿐, 실제 로봇 성공률과 같지는 않습니다. 실제 성공률은 rollout 평가로 측정합니다.

## 모델 보관

학습된 모델은 로컬에서 다음 폴더에 둡니다.

```text
models/act_lens_cap_v1/
```

이 폴더는 `model.safetensors` 등 대용량 가중치를 포함하므로 GitHub 일반 Git에는 올리지 않습니다. 원본은 Google Drive 또는 Hugging Face에 별도로 보관합니다.

## 자율 실행

자율 실행에는 팔로워와 카메라만 사용합니다. 실행 전에 카메라 구도, 상자·뚜껑 위치, 팔의 시작 자세를 학습 데이터 수집 시점과 최대한 일치시킵니다.

```bash
uv run lerobot-rollout --strategy.type=base --policy.path=/Users/doyounglim/Desktop/SO101/models/act_lens_cap_v1 --device=cpu --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B140307781 --robot.id=so101_follower --robot.max_relative_target=5 --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --task="Place the lens cap into the open box." --duration=30 --display_data=false
```

`--robot.max_relative_target=5`는 한 번에 너무 큰 관절 명령이 들어올 때 제한하는 안전장치입니다. 경고가 나오더라도 값을 높이거나 제거하지 않습니다.

## 자율 실행 평가

LeRobot 0.6.0에서는 `lerobot-record`가 리더암 시연 수집용입니다. 학습 정책 평가와 자동 영상 기록에는 `lerobot-rollout --strategy.type=episodic`을 사용합니다.

평가 데이터셋 이름은 반드시 `rollout_`으로 시작해야 합니다.

```bash
uv run lerobot-rollout --strategy.type=episodic --policy.path=/Users/doyounglim/Desktop/SO101/models/act_lens_cap_v1 --device=cpu --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B140307781 --robot.id=so101_follower --robot.max_relative_target=5 --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --task="Place the lens cap into the open box." --dataset.repo_id=DY-01/rollout_act_lens_cap_v1_mac --dataset.root=/Users/doyounglim/Desktop/SO101/lerobot_data/rollout_act_lens_cap_v1_mac --dataset.single_task="Place the lens cap into the open box." --dataset.fps=30 --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10 --dataset.video=true --dataset.push_to_hub=false --dataset.streaming_encoding=false --dataset.encoder_threads=1 --strategy.reset_to_initial_position=true --display_data=false
```

### 평가 진행 방식

1. 각 회차가 끝난 뒤 팔이 시작 자세로 돌아오면 상자와 뚜껑을 시작 위치에 되돌립니다.
2. 영상과 관절 로그가 별도 rollout 데이터셋에 저장됩니다.
3. 영상 검수 후 성공 여부를 표시합니다.
4. 성공률은 `성공 에피소드 수 / 완전한 평가 에피소드 수`로 계산합니다.

현재 CPU 추론 속도는 약 3~5Hz 경고가 발생할 수 있습니다. USB-C 허브 연결 카메라에서 프레임 시간초과가 발생할 수 있으므로 다른 USB 장치를 최소화하고, 평가 중에는 `--display_data=false`, `--dataset.streaming_encoding=false`를 사용합니다.

### 평가 중 키 조작

- `→`: 현재 에피소드를 즉시 끝내고 저장
- `←`: 현재 에피소드를 버리고 다시 기록
- `Esc`: 평가 세션 종료

`→`로 조기 종료한 에피소드는 부분 기록이므로 성공률 평가에서 제외하거나 삭제합니다.

부분 에피소드 삭제 예시:

```powershell
uv run lerobot-edit-dataset --repo_id=DY-01/rollout_act_lens_cap_v1 --root="<평가 데이터 폴더>" --new_repo_id=DY-01/rollout_act_lens_cap_v1 --new_root="<평가 데이터 폴더>" --operation.type=delete_episodes --operation.episode_indices="[9]" --push_to_hub=false
```

## 알려진 이슈와 대응

| 증상 | 원인 또는 상태 | 대응 |
|---|---|---|
| `OpenCVCamera(0) read failed` 또는 frame timeout | 허브 대역폭·전력 부족, 카메라 점유 또는 프레임 갱신 실패 | 카메라 앱을 종료하고 USB를 재연결합니다. 다른 USB 장치를 줄이거나 로봇과 카메라를 다른 포트/전원형 허브로 분리합니다. |
| `Record loop is running slower` | 로컬 CPU 추론이 목표 30Hz보다 느림 | 성능 평가에 기록하고, 향후 CUDA 가능 PC 또는 더 가벼운 실행 환경 검토 |
| `Relative goal position ... clamped` | 정책이 큰 관절 이동을 예측함 | `max_relative_target=5` 유지, 절대 제한을 풀지 않음 |
| `Could not write TorqueEnable on id=5` | wrist_roll 모터 통신/토크 설정이 일시 실패 | 12V·USB를 재연결하고 기존 캘리브레이션 파일로 연결 점검 |
| 서보 기어 파손 | 팔로워 3번 모터의 내부 기어 손상 | STS3215-12V 교체, ID 3 설정 및 재캘리브레이션 완료 |

## 다음 단계

1. macOS의 동일 조건에서 완전한 평가 10개를 새로 수집합니다.
2. 10개 영상을 검수해 성공·실패를 표시하고 ACT v1의 실제 성공률을 계산합니다.
3. 뚜껑 위치를 고정한 평가와 위치를 조금씩 바꾼 평가를 구분합니다.
4. 실패 양상을 분류합니다.
   - 시작 자세·카메라 구도 불일치
   - 느린 CPU 추론
   - 집기 실패
   - 상자 위치 오차
   - 과도한 관절 명령 및 안전 제한
5. 성공률이 낮으면 같은 환경에서 100~200개 수준으로 시연 데이터를 확장하고 `act_lens_cap_v2`를 재학습합니다.
6. 같은 10회 조건으로 v1과 v2를 비교합니다.

## GitHub에 포함하지 않는 항목

다음 항목은 `.gitignore`로 제외합니다.

- `.venv/`
- `models/`의 대용량 모델 가중치
- `outputs/`의 임시 카메라 이미지
- 원본 동영상 및 로컬 데이터셋
- Hugging Face 토큰, `.env`, 기타 비밀 정보

GitHub에는 코드, 의존성 파일, 실행·학습 문서, 결과 요약만 보관합니다.
