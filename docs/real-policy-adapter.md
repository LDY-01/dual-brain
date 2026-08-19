# 실물 SO-101 실행 어댑터와 그림자 모드

## 목적과 현재 상태

`kwon_lab/hardware/real_so101_adapter.py`는 시뮬레이션 정책이 사용하는 6관절 라디안 관측·액션과 실제 LeRobot SO-101을 연결한다. 손목 U20CAM 영상은 왜곡 없이 중앙 4:3 영역을 잘라 ACT v2.2의 640×480 입력으로 만들고, 상단 U20CAM의 1280×720 영상은 위치·자세 판정용으로 원본 비율을 유지한다.

현재 구현과 가짜 장비 회귀시험은 완료했지만 새 장소 실물 검증은 하지 않았다. 로봇과 카메라가 분리된 상태에서는 실물 관절 방향·영점과 프레임을 확인할 수 없으므로 능동 모드는 의도적으로 차단되어 있다.

| 구분 | 그림자 모드 | 능동 모드 |
|---|---|---|
| 실제 관절 읽기 | 예 | 예 |
| 손목·상단 영상 읽기 | 예 | 예 |
| ACT v2.2 추론 | 예 | 예 |
| 라디안→실물 단위 변환 | 계산·기록 | 계산·적용 |
| 상대·절대 안전 제한 | 계산·기록 | 계산·적용 |
| 모터 설정 레지스터 변경 | 아니요 | LeRobot 정상 연결 절차 사용 |
| `send_action`/Goal Position | **0회** | 모든 게이트 통과 후에만 가능 |

그림자 모드는 SO-101 직렬 버스를 읽기 전용으로 열어 `Present_Position`만 읽는다. 일반 `SO101Follower.connect()`가 수행하는 모터 설정을 호출하지 않으며, 종료할 때 토크 레지스터도 변경하지 않는다. 따라서 팔은 움직이지 않는다.

## 관절 단위 변환

ACT v2.2는 MuJoCo 관절 라디안을 사용한다. LeRobot은 팔 다섯 관절을 보정된 도 단위로, 그리퍼를 0~100 범위로 반환한다.

- 팔 다섯 관절 초기식: `robot_deg = policy_rad × 180/π`
- 그리퍼 초기식: MuJoCo 범위 -10~100도를 LeRobot 0~100에 선형 대응
- 실제 입력은 역변환해 ACT의 `observation.state`에 전달

이 식은 모델 정의로부터 얻은 초기 대응일 뿐 실제 조립 영점과 방향을 아직 확인하지 않았다. `config/real_robot_adapter.local.json`의 여섯 `verified_on_physical_robot` 값은 모두 `false`이며, 사용자가 새 장소에서 제안 방향과 실제 자세를 확인하기 전에는 바꾸지 않는다. 하나라도 `false`면 능동 모드는 차단된다.

## 오프라인 자체검증

```powershell
$env:PYTHONPATH='kwon_lab'
.\.venv\Scripts\python.exe kwon_lab/tools/real_policy_shadow.py `
  --adapter-config config/real_robot_adapter.example.json `
  --self-test
```

검증 항목은 관절 변환 왕복, 영상 크롭, 그림자 전송 0회, 읽기 전용 종료, 감사 로그, 프리플라이트 차단과 테스트용 능동 전송·안전 클램프다. 실제 장비에는 연결하지 않는다.

## 새 장소에서 실행할 그림자 시험

로봇 포트와 손목·상단 카메라를 연결하고 카메라 역할을 다시 확인한 뒤 다음을 실행한다.

```powershell
$env:PYTHONPATH='kwon_lab'
.\.venv\Scripts\python.exe kwon_lab/tools/real_policy_shadow.py `
  --adapter-config config/real_robot_adapter.local.json `
  --camera-config config/real_camera_roles.local.json `
  --safety-config config/real_robot_safety_limits.local.json `
  --calibration-config config/overhead_camera_calibration.local.json `
  --workspace-config config/real_workspace.local.json `
  --device cpu `
  --cycles 25
```

상단 보정이나 숄더팬 한계가 아직 없어도 그림자 실행은 가능하지만 보고서에는 `motion_authorized=false`와 누락 항목이 남는다. 출력은 Git 비추적 `results/real_policy_shadow/<시각>/`에 저장한다.

- `actions.jsonl`: 정책 라디안, 변환 전후 실물 목표, 상대·절대 제한 개입
- `summary.json`: Goal Position 0회 확인, 프리플라이트·매핑 상태, 장면 판정 집계
- `wrist_*.jpg`, `overhead_*.jpg`: 주기적 입력 증거

그림자 결과는 동작 방향과 소프트웨어 연결을 검사할 뿐 실제 집기 성공률을 뜻하지 않는다. 제안 관절 방향, 카메라 역할, 프레임 색상과 좌표가 정상임을 검수한 다음 관절 매핑 확인 절차와 저속 최초 픽으로 넘어간다.

## 능동 모드의 차단 조건

실제 명령 경로는 코드에 있지만 일반 실행 도구로 아직 노출하지 않았다. 다음 세 조건이 모두 충족되지 않으면 `RealSO101Adapter(mode="active")` 생성 단계에서 실패한다.

1. 새 장소 프리플라이트의 `motion_authorized=true`
2. 여섯 관절 매핑의 `verified_on_physical_robot=true`
3. 별도 능동 실행 확인 토큰 일치

실물 그림자 검증 결과를 사람이 확인하기 전에는 능동 실행 도구를 열지 않는다.
