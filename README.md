# SO101 렌즈 뚜껑 넣기 프로젝트

SO101 리더-팔로워 로봇을 사용해 **렌즈 뚜껑을 열린 상자 안에 넣는 작업**을 시연 데이터로 학습시키고, 학습된 ACT 정책으로 팔로워암을 자율 실행하는 프로젝트입니다.

> 이 문서는 2026-07-31 기준의 작업 기록과 재현 절차입니다. 실제 로봇을 움직이기 전에 항상 주변을 비우고, 비정상 진동이나 충돌이 발생하면 팔로워의 12V 전원을 즉시 분리합니다.

## 현재 상태

- SO101 리더암·팔로워암 조립, USB 포트 식별, 캘리브레이션 완료
- Innomaker U20CAM-720P 카메라 연결 및 학습 장면 고정 완료
- 초기 시연 데이터셋 `version1` 수집·검수 완료
- ACT 모델 `act_lens_cap_v1` 학습 완료
- 팔로워암 단독 자율 실행(rollout) 확인 완료
- 실행 환경을 Windows에서 macOS로 이전
  - follower 포트: `/dev/tty.usbmodem5B140307781`
  - leader 포트: `/dev/tty.usbmodem5B3D0456881`
  - 전방 카메라: OpenCV index `0`
  - Windows에서 사용한 follower 캘리브레이션 파일 이전 완료
- macOS 고정 조건 정식 rollout 평가 10회 완료
  - 최종 성공: 2회 (2회차, 7회차)
  - 최종 실패: 8회
  - 성공률: 20%
  - 모든 회차에서 뚜껑 접근은 수행함
  - 뚜껑 집기 성공: 4회
  - 집기에 성공한 4회 중 상자 투입 성공: 2회
- 집기 실패를 보강하기 위한 성공 시연 데이터셋 `version2` 수집 완료
  - 로컬 데이터셋: `DY-01/lens_cap_into_box_version2`
  - 50개 에피소드, 45,000 프레임, 30 FPS
  - 뚜껑 위치를 작업 영역 안에서 무작위로 바꿔 수집
  - 텔레오퍼레이션 수집 시 `robot.max_relative_target=10` 사용
- `version1`과 `version2`를 통합한 학습 데이터셋 `v1+v2` 생성 완료
  - 업로드용 ID: `DY-01/lens_cap_into_box_v1_v2`
  - 100개 에피소드, 90,000 프레임, 30 FPS
  - Hugging Face 업로드 완료
- ACT 모델 `act_lens_cap_v2` 학습 완료
  - Google Colab Tesla T4에서 40,000 step 학습
  - 최종 checkpoint 저장 완료
- `act_lens_cap_v2` 고정 조건 정식 rollout 평가 10회 완료
  - 최종 성공: 2회
  - 최종 실패: 8회
  - 성공률: 20% (v1과 동일)
  - 평가 데이터: `rollout_act_lens_cap_v2_mac`, 8,529 프레임

현재 v1과 v2 모두 고정 조건 정식 평가에서 20% 성공률을 기록했습니다. 데이터 수를 100개로 늘린 것만으로는 집기 병목이 해소되지 않았으므로, 다음 단계의 목적은 **v2 실패 영상을 분석해 보강할 동작 품질과 조건을 좁히는 것**입니다.

## 하드웨어 구성

| 구성품 | 역할 | 현재 macOS 설정 |
|---|---|---|
| SO101 follower | 자율 실행 팔 | `/dev/tty.usbmodem5B140307781`, 별도 12V 전원 |
| SO101 leader | 시연 데이터 수집용 팔 | `/dev/tty.usbmodem5B3D0456881` (자율 실행에는 불필요) |
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

리더는 macOS에서 포트를 확인하고 2026-07-31에 새로 캘리브레이션했습니다.

```text
leader port:
/dev/tty.usbmodem5B3D0456881

leader calibration:
/Users/doyounglim/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so101_leader.json
```

실행 중 기존 캘리브레이션을 사용할지 묻는 경우 `Enter`를 누릅니다. `c`를 누르면 해당 파일을 새 캘리브레이션으로 덮어쓰므로, 모터 교체나 조립 변경처럼 재캘리브레이션이 필요한 경우가 아니면 누르지 않습니다.

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
| 데이터셋 이름 | `version1` |
| 로컬 경로 | `lerobot_data/lens_cap_into_box_version1` |
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

### 집기 보강 데이터

| 항목 | 값 |
|---|---|
| 데이터셋 이름 | `version2` |
| 로컬 경로 | `lerobot_data/lens_cap_into_box_version2` |
| 에피소드 수 | 50 |
| 총 프레임 | 45,000 |
| 에피소드 길이 | 30초 |
| 목표 FPS | 30 |
| 뚜껑 배치 | 표시된 작업 영역 내 무작위 위치 |

`version1`과 같은 카메라 해상도, 관절 상태, 행동 형식으로 저장됐습니다.

### 통합 학습 데이터

| 항목 | 값 |
|---|---|
| 데이터셋 이름 | `v1+v2` |
| 업로드용 ID | `DY-01/lens_cap_into_box_v1_v2` |
| Hugging Face | `https://huggingface.co/datasets/DY-01/lens_cap_into_box_v1_v2` |
| LeRobot 데이터셋 태그 | `v3.0` |
| 로컬 경로 | `lerobot_data/lens_cap_into_box_v1_v2` |
| 에피소드 수 | 100 |
| 총 프레임 | 90,000 |
| FPS | 30 |

Hugging Face 리포지토리 이름에는 `+`를 쓸 수 없으므로, 업로드할 때는 `v1_v2`를 사용합니다.

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
  --dataset.repo_id=DY-01/lens_cap_into_box_version1 \
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

### 홈 자세 복귀 원칙

앞으로 작성하거나 수정하는 모든 자율 실행·평가 코드는 rollout이 성공, 실패, 시간 제한 종료 중 어느 경우로 끝나더라도 **정책을 먼저 정지한 뒤 팔로워를 안전한 속도로 홈 자세로 복귀**시켜야 합니다. 홈 자세는 캘리브레이션의 0점이 아니라, 이 작업을 시작할 때 사용하는 사용자 지정 관절 자세입니다.

홈 복귀는 ACT 정책이 학습해서 수행할 동작이 아니라 별도 제어 로직으로 처리합니다. 통신 오류, 충돌 위험, 비정상 진동 또는 비상 정지가 감지된 경우에는 홈 복귀를 시도하지 않고 즉시 모터를 정지하거나 12V 전원을 분리합니다. 홈 관절값은 실제 로봇에서 충돌 없이 확인한 뒤 코드 설정값으로 관리합니다.

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

1. 각 회차가 끝나면 정책을 정지하고, 별도 홈 복귀 제어로 팔을 안전한 시작 자세에 되돌린 뒤 상자와 뚜껑을 시작 위치에 되돌립니다.
2. 영상과 관절 로그가 별도 rollout 데이터셋에 저장됩니다.
3. 영상 검수 후 성공 여부를 표시합니다.
4. 성공률은 `성공 에피소드 수 / 완전한 평가 에피소드 수`로 계산합니다.

현재 CPU 추론 속도는 약 3~5Hz 경고가 발생할 수 있습니다. USB-C 허브 연결 카메라에서 프레임 시간초과가 발생할 수 있으므로 다른 USB 장치를 최소화하고, 평가 중에는 `--display_data=false`, `--dataset.streaming_encoding=false`를 사용합니다.

### ACT v1 정식 평가 결과

2026-07-31에 macOS의 고정된 카메라·상자·뚜껑 위치에서 10개 에피소드를 평가했습니다.

| 회차 | 뚜껑 접근 | 뚜껑 집기 | 상자 접근 | 상자 투입 | 최종 결과 |
|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | O | X | - | - | 실패 |
| 2 | O | O | O | O | 성공 |
| 3 | O | O | O | X | 실패 |
| 4 | O | O | O | X | 실패 |
| 5 | O | X | - | - | 실패 |
| 6 | O | X | - | - | 실패 |
| 7 | O | O | O | O | 성공 |
| 8 | O | X | - | - | 실패 |
| 9 | O | X | - | - | 실패 |
| 10 | O | X | - | - | 실패 |

- 전체 성공률: `2 / 10 = 20%`
- 뚜껑 접근 성공률: `10 / 10 = 100%`
- 뚜껑 집기 성공률: `4 / 10 = 40%`
- 집기 성공 후 상자 투입 성공률: `2 / 4 = 50%`
- 평가 데이터: `lerobot_data/rollout_act_lens_cap_v1_mac`
- 저장 결과: 10개 에피소드, 8,567 프레임

가장 큰 병목은 뚜껑 집기 단계입니다. 먼저 실패 영상에서 그리퍼의 접근 위치, 높이, 닫는 시점을 비교하고, 해당 실패 조건을 중심으로 보강 데이터를 수집합니다.

### ACT v2 정식 평가 결과

2026-07-31에 v1과 같은 고정된 카메라·상자·뚜껑 위치에서 10개 에피소드를 평가했습니다.

- 전체 성공률: `2 / 10 = 20%`
- 평가 데이터: `lerobot_data/rollout_act_lens_cap_v2_mac`
- 저장 결과: 10개 에피소드, 8,529 프레임

v2는 100개 성공 시연으로 학습했지만 v1의 20%를 넘지 못했습니다. 따라서 다음 보강에서는 무작위 위치를 더 넓히기 전에, 성공·실패 영상에서 그리퍼 중심 위치, 접근 높이, 닫는 시점, 파지 후 들어 올림을 비교해 실패 원인을 먼저 특정합니다.

### ACT v2 홈 자세 평가 결과

2026-08-04에 실제 태스크 시작 자세를 관절값으로 저장하고, 각 에피소드 종료 후 별도 제어로 홈 자세에 복귀하는 조건에서 v2를 10회 평가했습니다.

- 전체 성공률: `4 / 10 = 40%`
- 평가 데이터: `lerobot_data/rollout_act_lens_cap_v2_home_mac`
- 홈 자세 설정: `config/lens_cap_home_pose.json`

홈 자세를 고정한 뒤 성공률은 기존 20%에서 40%로 상승했습니다. 다만 표본이 10회이므로, 이 결과만으로 데이터 보강 효과와 홈 자세 효과를 분리해 확정하지는 않습니다.

관찰된 실패 원인:

1. 뚜껑의 위치를 정확히 중심에 맞춰 집기보다, 보강 데이터에서 자주 나타난 접근 위치 중 하나로 향하는 양상이 보입니다. 넓은 무작위 위치를 50개 시연으로만 수집하면 위치별 예시가 부족해질 수 있습니다.
2. 그리퍼가 뚜껑을 밀어 위치가 바뀐 뒤에도 새 카메라 관측에 맞춰 재탐색·재접근하지 못하고, 원래 접근 동작을 계속하는 양상이 보입니다. 기존 데이터에는 이런 복구 상태에서 최종 성공으로 이어지는 시연이 없습니다.

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
| `Relative goal position ... clamped` | 목표 관절 위치와 현재 위치 차이가 안전 제한보다 큼 | rollout에서는 `max_relative_target=5`를 유지합니다. 텔레오퍼레이션 수집에서는 리더·팔로워 시작 자세를 맞춘 뒤 `10`을 사용했습니다. |
| `Could not write TorqueEnable on id=5` | wrist_roll 모터 통신/토크 설정이 일시 실패 | 12V·USB를 재연결하고 기존 캘리브레이션 파일로 연결 점검 |
| 서보 기어 파손 | 팔로워 3번 모터의 내부 기어 손상 | STS3215-12V 교체, ID 3 설정 및 재캘리브레이션 완료 |

## Isaac Sim + Isaac Lab 전환 계획

실물 로봇으로 실패 상태를 하나씩 시연하는 대신, NVIDIA Isaac Sim과 Isaac Lab을 사용해 합성 데이터를 대량 생성하고 sim-to-real 전이를 수행합니다. Isaac Sim은 장면·물리·카메라 렌더링을 담당하고, Isaac Lab은 병렬 환경에서 데이터 생성과 정책 학습을 담당합니다.

현재 `simulation_assets/so101`에는 SO101의 MuJoCo(MJCF) 및 URDF 자산이 있습니다. Isaac 환경에는 이 로봇 자산을 USD/URDF 기반으로 가져오고, 상자·렌즈 뚜껑·책상·고정 RGB 카메라를 추가로 구성해야 합니다. 현재 자산은 그리퍼의 LeRobot 표현(`0=닫힘`, `100=열림`)과 동일한 제어 매핑이 아직 반영되지 않았으므로, 집기 학습 전에 이를 검증합니다.

### 시뮬레이션 데이터 생성 범위

1. 뚜껑을 화면 내 격자와 연속 좌표에 배치하고, 집기-운반-상자 투입까지 성공하는 궤적을 자동 생성합니다.
2. 뚜껑이 그리퍼에 밀려 좌·우·위·아래로 벗어난 상태를 의도적으로 만들고, 재탐색-재접근-재그립 후 성공하는 복구 궤적을 생성합니다.
3. 뚜껑·상자 위치, 크기, 마찰, 질량, 그리퍼 접촉, 책상 재질을 랜덤화합니다.
4. 조명, 그림자, 배경, 색상, 카메라 자세와 내부 파라미터, 이미지 노이즈를 랜덤화해 실제 웹캠과의 시각적 차이를 줄입니다.
5. 상태 기반 IK 또는 스크립트 전문가 정책으로 성공 궤적을 생성하고, 렌더링된 RGB 영상과 관절 행동을 LeRobot 호환 데이터셋 형태로 내보냅니다.

### 학습 및 실제 로봇 검증

1. 합성 데이터로 뚜껑 위치 인식과 복구 행동을 먼저 학습합니다.
2. 실제 로봇에서는 카메라 구도·물체 치수·그리퍼 접촉 차이를 보정하기 위한 소량의 고품질 시연만 수집합니다.
3. 합성 데이터와 실제 보정 데이터를 병합 또는 순차 미세 조정하고, 홈 자세 고정 조건에서 정식 10회 평가를 반복합니다.
4. 실제 실행에서는 `robot.max_relative_target=5`와 별도 홈 복귀 제어를 유지합니다.

### 실행 환경

Isaac Sim과 Isaac Lab 본체는 NVIDIA RTX GPU가 있는 Windows 또는 Linux 워크스테이션/서버에서 실행합니다. 현재 macOS 장비는 실제 SO101 연결·평가와 원격 시뮬레이터 화면 확인 용도로 사용합니다. 공식 요구사항은 RTX 4080급 16GB VRAM, RAM 32GB, SSD 여유 공간 50GB 이상을 최소 기준으로 제시합니다.

- [NVIDIA SO-101 sim-to-real 도메인 랜덤화 가이드](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html)
- [Isaac Lab 시각 도메인 랜덤화 가이드](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/transferring-robot-learning-policies-from-simulation-to-reality/03-bridging-the-gap-simulation-enhancement/01-visual-domain-randomization.html)
- [Isaac Sim 공식 요구사항](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)

## 다음 단계

1. 사용할 Windows/Linux NVIDIA RTX GPU 워크스테이션 또는 클라우드 환경을 확보합니다.
2. SO101, 상자, 렌즈 뚜껑, 책상, 카메라를 포함한 Isaac Sim 장면을 구성하고 실제 치수·카메라 구도를 맞춥니다.
3. 그리퍼 제어와 물리 접촉을 검증한 뒤, 성공 및 복구 궤적을 자동 생성합니다.
4. 도메인 랜덤화 합성 데이터로 정책을 학습하고, 실제 데이터로 소량 보정합니다.
5. 홈 자세 고정 조건에서 실제 로봇 정식 10회 평가를 수행하고 v1·v2·시뮬레이션 보정 모델을 비교합니다.

## GitHub에 포함하지 않는 항목

다음 항목은 `.gitignore`로 제외합니다.

- `.venv/`
- `models/`의 대용량 모델 가중치
- `outputs/`의 임시 카메라 이미지
- 원본 동영상 및 로컬 데이터셋
- Hugging Face 토큰, `.env`, 기타 비밀 정보

GitHub에는 코드, 의존성 파일, 실행·학습 문서, 결과 요약만 보관합니다.
