# ADAS Project

## Setup

```bash
# 카메라 설정 도구 설치
sudo apt install -y v4l-utils

# 의존성 설치 및 가상환경 동기화
uv sync
```

## Run

```bash
# 카메라 + YOLO 추적 실행
uv run python main.py

# 카메라 + YOLO 추적 + Dynamixel 모터 제어
uv run python main.py --motor

# 라즈베리파이 영상 스트림 기반 웹 서버 실행
uv run python local.py
```

## Test

```bash
# 얼굴 검출/방향 표시 테스트
uv run python test/face_tracker_test.py

# 얼굴 검출/방향 표시 + Dynamixel 모터 제어 테스트
uv run python test/face_tracker_test.py --motor

# 모터 좌표 입력 테스트(DRY_RUN)
uv run python test/motor_test.py

# 실제 Dynamixel 모터 좌표 입력 테스트
uv run python test/motor_test.py --motor
```
