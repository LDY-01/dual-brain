# 실물 상단 카메라 보정 절차

## 목적

상단 U20CAM-720P의 픽셀 좌표를 SO-101 베이스 기준 테이블 XY 좌표(m)로 변환한다. 실물 설정값은 `config/overhead_camera_calibration.local.json`에 저장되며 Git 추적 대상이 아니다.

## 설치 기준

1. 렌즈와 테이블 사이 거리를 약 52cm로 맞춘다.
2. 카메라 광축이 테이블을 거의 수직으로 보도록 고정한다.
3. SO-101의 shoulder-pan 회전축을 테이블에 수직으로 내린 점을 XY 원점으로 사용한다.
4. 로봇에서 앞쪽을 +X, 로봇이 바라보는 기준 왼쪽을 +Y로 사용한다.
5. 전체 1.4×0.6m 테이블이 아니라 로봇이 닿는 약 X=18~28cm, Y=-10~10cm 영역이 선명하게 보이면 된다.

카메라 설치 후에는 브래킷, 해상도, 줌, 초점을 바꾸지 않는다. 하나라도 바꾸면 다시 보정한다.

## 카메라 역할 등록

2026-08-19 현재 Windows PnP 목록에는 외부 `720p HD Camera` 한 대가 보이지만 손목·상단 역할은 모두 미등록 상태다. OpenCV index만으로 역할을 확정하지 않으며 PnP 장치 ID, 물리 USB 포트 라벨, 사람이 확인한 화면을 함께 등록한다. 설정 파일은 `config/real_camera_roles.local.json`이며 장비별 정보이므로 Git에 포함하지 않는다.

현재 상태 확인:

```powershell
$env:PYTHONPATH='kwon_lab'
.\.venv\Scripts\python.exe kwon_lab/tools/register_real_cameras.py `
  --status --config config/real_camera_roles.local.json
```

상단 카메라를 고정·연결한 뒤 먼저 사용 가능한 index를 탐색한다.

```powershell
.\.venv\Scripts\python.exe kwon_lab/tools/register_real_cameras.py `
  --probe --max-index 5 --skip-index 0
```

화면을 직접 확인해 테이블을 수직으로 보는 index를 찾은 뒤 등록한다. `<USB_PORT_LABEL>`에는 실제로 고정한 허브·포트 이름을 적는다.

```powershell
.\.venv\Scripts\python.exe kwon_lab/tools/register_real_cameras.py `
  --register-role overhead `
  --camera-index <상단_INDEX> `
  --physical-usb-port <USB_PORT_LABEL> `
  --confirm-view OVERHEAD
```

손목과 상단에 같은 index를 등록하면 도구가 거부한다. 향후 실물 제어 시작부는 `require_dual_camera_ready()`를 호출해 두 카메라가 모두 등록되고 실제 프레임을 반환할 때만 모터 동작을 허용한다.

## 기준점 표시

`config/overhead_calibration_points.example.json`의 6개 XY 위치를 줄자와 직각자를 이용해 테이블에 표시한다. 점의 중심 오차가 그대로 로봇 집기 오차가 되므로 가능한 한 2mm 이내로 표시한다.

## 실행

현재 Windows index 0은 노트북 내장 카메라다. 손목·상단 U20CAM을 동시에 연결한 뒤 장치별 인덱스를 다시 확인하고, 화면을 직접 확인해 그리퍼가 보이는 카메라를 손목으로, 작업대를 수직으로 보는 카메라를 상단으로 등록한다. 확인된 상단 카메라 인덱스로 다음 명령을 실행한다.

```powershell
$env:PYTHONPATH='kwon_lab'
.\.venv\Scripts\python.exe kwon_lab/tools/calibrate_overhead_camera.py `
  --camera-index <확인된_상단카메라_인덱스> `
  --backend dshow `
  --width 1280 `
  --height 720 `
  --lens-height-m 0.52 `
  --layout-id <NEW_WORKSPACE_LAYOUT_ID> `
  --plane target_table `
  --output config/overhead_camera_calibration.local.json
```

화면 상단에 표시되는 이름 순서대로 기준점 중심을 클릭한다. 잘못 클릭하면 `R`, 모두 클릭했으면 `Enter`, 취소는 `Esc`다.

테이블면을 한 번 보정하면 52cm 렌즈 높이를 사용해 서 있는 블록의 6cm 상단면과 쓰러진 블록의 4cm 상단면 변환도 자동 생성한다. 이는 카메라가 거의 수직이라는 전제의 빠른 MVP 보정이다. 카메라가 기울었거나 오차가 크면 높이별 평면을 직접 보정해 같은 이름으로 덮어써야 한다.

## 통과 기준

- 최대 기준점 오차 10mm 이하는 최소 통과
- 권장 최대 오차 5mm 이하
- 보정 후 알려진 세 위치에 블록을 놓아 추정 XY를 다시 확인
- 실제 픽은 저속·낮은 힘·비상정지 준비 상태에서 시작

보정 파일에 필요한 행렬은 `target_table`, `upright_top_6cm`, `tipped_top_4cm` 세 평면으로 저장된다.
