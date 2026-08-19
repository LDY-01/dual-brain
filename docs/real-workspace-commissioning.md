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

`--skip-live-camera`는 진단용이며 항상 `motion_authorized=false`로 끝난다. 이 도구 자체는 모터를 제어하지 않는다.

## 5. 최초 픽 10회 기록

```powershell
.\.venv\Scripts\python.exe kwon_lab/tools/real_first_pick_trials.py
```

이 도구도 모터를 제어하지 않는다. 프리플라이트 통과 후 별도로 승인된 저속 제어기로 한 번씩 픽하고, 도구가 두 카메라의 전후 프레임·블록 자세·집기 성공·리프트 확인·실패 단계를 저장한다. 10회 중 8회 이상이면 저속 PLACE 검증으로 진행하고, 미만이면 실패 영상을 먼저 분류한 뒤 물성 무작위화 파인튜닝 여부를 결정한다.

## 현재 상태

2026-08-19 기준 노트북 내장 카메라는 index 0, 손목 U20CAM은 index 1, 상단 U20CAM은 index 2로 확인했다. 동일 모델 두 대는 손목 카메라를 한 번 분리해 PnP ID를 구분했고 두 역할의 1280×720 실시간 검사가 통과했다. 새 장소 workspace 파일의 현장 확인값은 모두 `false`이며 숄더팬 한계와 호모그래피도 아직 없으므로 실물 동작은 정상적으로 차단된다. 장소 이전 뒤 index·PnP ID 검사를 다시 실행한다.
