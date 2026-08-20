# 새 작업공간 실물 전환 절차

이 절차는 장소가 바뀐 뒤 카메라·기둥·로봇 배치가 확정되지 않은 상태에서 실물 모터가 우발적으로 시작되는 것을 막는다. 모든 로컬 장비 설정은 `config/*.local.json`, 원시 영상과 세션 결과는 `results/`에 저장되어 Git에 포함되지 않는다.

## 1. 카메라 식별

현재 OpenCV index는 USB 재연결 후 바뀔 수 있다. 따라서 index, Windows PnP 장치 ID, 실제 USB 포트 라벨, 사람이 확인한 화면을 함께 기록한다. 동일한 U20CAM 두 대의 PnP ID를 구분하기 어려우면 한 대씩 연결해 `--devices`를 실행한다.

```powershell
$env:PYTHONPATH='kwon_lab'
.\.venv\Scripts\python.exe kwon_lab/tools/register_real_cameras.py --devices
.\.venv\Scripts\python.exe kwon_lab/tools/register_real_cameras.py `
  --probe --max-index 5 `
  --snapshot-dir results/real_camera_probe/new_workspace
```

저장된 `camera_index_*.png`를 직접 보고 역할을 확인한 뒤 장치 ID와 물리 포트를 함께 등록한다.

```powershell
.\.venv\Scripts\python.exe kwon_lab/tools/register_real_cameras.py `
  --register-role wrist --camera-index <WRIST_INDEX> `
  --device-instance-id '<WRIST_PNP_ID>' `
  --physical-usb-port WRIST_FIXED_PORT --confirm-view WRIST

.\.venv\Scripts\python.exe kwon_lab/tools/register_real_cameras.py `
  --register-role overhead --camera-index <OVERHEAD_INDEX> `
  --device-instance-id '<OVERHEAD_PNP_ID>' `
  --physical-usb-port OVERHEAD_FIXED_PORT --confirm-view OVERHEAD
```

장치 ID가 현재 Windows 목록에 없거나, 두 역할이 같은 index/장치 ID를 쓰거나, 물리 포트 라벨이 없으면 준비 완료로 인정하지 않는다.

## 2. 새 배치 ID와 현장 확인

`config/real_workspace.example.json`을 `config/real_workspace.local.json`으로 복사하고 새 배치에 고유한 `layout_id`를 준다. 로봇 베이스·카메라 기둥을 고정하고 비상정지·작업영역을 실제로 확인한 항목만 `true`로 바꾼다. 현행 작업대 표면은 `white_matte_ceramic`이다.

기둥 반대편에 실제로 사용할 X/Y 작업영역을 정하고 테이프로 표시한다. 블록 시작 위치와 목표뿐 아니라 경계 복구용 2cm 여유까지 상단 카메라 화면과 보정점 볼록껍질 안에 있어야 한다. 이 확인을 마친 뒤에만 `task_workspace_marked=true`로 바꾼다.

## 3. 같은 배치 ID로 안전 한계와 카메라 보정

다음 두 명령에는 workspace 파일과 정확히 같은 `<LAYOUT_ID>`를 사용한다.

```powershell
.\.venv\Scripts\python.exe kwon_lab/tools/capture_shoulder_pan_limit.py `
  --port <SO101_PORT> --layout-id <LAYOUT_ID>

.\.venv\Scripts\python.exe kwon_lab/tools/calibrate_overhead_camera.py `
  --camera-index <OVERHEAD_INDEX> --backend dshow `
  --width 1280 --height 720 --lens-height-m <MEASURED_HEIGHT_M> `
  --layout-id <LAYOUT_ID> --plane target_table
```

숄더팬 한계는 먼저 저장하지 않고 확인한 뒤 같은 명령에 `--save`를 붙인다. 보정은 최대 기준점 오차가 10mm 이하여야 하며 5mm 이하를 권장한다. 배치 ID가 다르면 값이 모두 존재해도 프리플라이트가 차단한다.

## 4. 통합 프리플라이트

```powershell
.\.venv\Scripts\python.exe kwon_lab/tools/real_workspace_preflight.py
```

다음을 모두 통과해야 `motion_authorized=true`가 된다.

- 손목·상단 카메라의 index, PnP ID, USB 포트 라벨과 실시간 프레임
- 고정된 새 배치의 숄더팬 절대 한계
- 같은 배치에서 만든 상단 카메라 3개 높이 평면 보정
- 흰색 무광 세라믹 작업대 확인
- 로봇·기둥 고정, 작업영역 정리, 비상정지 준비, 두 화면 육안 확인
- 실제 작업영역·목표·2cm 복구 여유의 테이프 표시와 상단 화면 포함

`--skip-live-camera`는 진단용이며 항상 `motion_authorized=false`로 끝난다. 이 도구 자체는 모터를 제어하지 않는다.

## 5. 최초 픽 10회 기록

```powershell
.\.venv\Scripts\python.exe kwon_lab/tools/real_first_pick_trials.py
```

이 도구도 모터를 제어하지 않는다. 프리플라이트 통과 후 별도로 승인된 저속 제어기로 한 번씩 픽하고, 도구가 두 카메라의 전후 프레임·블록 자세·집기 성공·리프트 확인·실패 단계를 저장한다.

- 9–10/10: 저속 전체 픽앤플레이스로 진행
- 8/10: 동일 조건 10회 추가 후 다시 판정
- 0–7/10: 일반 속도로 가지 않고 보정·인식·그리퍼·정책 실패를 먼저 분류

## 6. 저속 전체 작업과 단계적 속도 상승

최초 픽 게이트 통과 후 저속 전체 픽앤플레이스 10회를 실행한다. 각 회차에서 최초 집기, 운반 낙하, 재탐색·재집기, 6cm 놓기, 75% 포함, 작업시간과 재시도 횟수를 기록한다.

저속 안전성과 복구가 확인된 뒤에만 `50% → 70% → 100%` 순서로 속도를 높인다. 각 단계에서 같은 지표를 다시 확인하며, 충돌·비정상 진동·반복 낙하·안전 제한 초과가 한 번이라도 있으면 즉시 전 단계로 돌아간다. 실물 실패는 먼저 보정·인식·하드웨어·상태 머신 문제와 정책 문제로 분리하고, 반복되는 정책 실패 구간만 파인튜닝 데이터로 사용한다.

## 현재 상태

2026-08-19 기준 노트북 내장 카메라는 index 0, 손목 U20CAM은 index 1, 상단 U20CAM은 index 2로 확인했다. 동일 모델 두 대는 손목 카메라를 한 번 분리해 PnP ID를 구분했고 두 역할의 1280×720 실시간 검사가 통과했다. 2026-08-20 시뮬레이션 소프트웨어 게이트는 정상 45/50(90%)로 통과했다. 새 장소 workspace 파일의 현장 확인값은 모두 `false`이며 작업영역 표시·숄더팬 한계·호모그래피도 아직 없으므로 실물 동작은 정상적으로 차단된다. 장소 이전 뒤 index·PnP ID 검사를 다시 실행하고 읽기 전용 그림자 모드부터 시작한다.
