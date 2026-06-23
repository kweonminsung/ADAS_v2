#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
얼굴 추적 테스트 버전

- 웹캠으로 얼굴을 검출하고 코 위치를 추정
- 화면 중심 기준으로 방향(LEFT/RIGHT/UP/DOWN) 표시
- Dynamixel 모터가 연결돼 있으면 face_tracker_xl430_06_14.Tracker 로 신호 전송
- 모터 없이도 방향 확인만 가능 (DRY_RUN 모드)

실행:
  python3 face_tracker_test.py            # 모터 없이 방향만 확인
  python3 face_tracker_test.py --motor    # 모터 연결 + 제어 포함
"""

import argparse
import os
import sys

# oCam 등 비표준 포맷 카메라를 OpenCV로 열기 위해 libv4l2 변환 라이브러리를 자동 적용
_V4L2_LIB = "/usr/lib/x86_64-linux-gnu/libv4l/v4l2convert.so"
if os.path.exists(_V4L2_LIB) and "LD_PRELOAD" not in os.environ:
    os.environ["LD_PRELOAD"] = _V4L2_LIB
    os.execv(sys.executable, [sys.executable] + sys.argv)

import cv2

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
FRAME_W, FRAME_H   = 1280, 720
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2
DEADZONE_PX        = 20   # 이 픽셀 이내 오차는 "중앙"으로 간주

# 얼굴 bbox 안에서 코 위치 비율 (수직: 위에서 55%, 수평: 중앙)
NOSE_Y_RATIO = 0.55


# ──────────────────────────────────────────────
# 코 좌표 추정 (얼굴 bbox → 코 중심)
# ──────────────────────────────────────────────

def estimate_nose(x, y, w, h):
    nose_x = x + w // 2
    nose_y = y + int(h * NOSE_Y_RATIO)
    return nose_x, nose_y


# ──────────────────────────────────────────────
# 방향 판단
# ──────────────────────────────────────────────

def get_direction(nose_x, nose_y):
    parts = []
    err_x = CENTER_X - nose_x
    err_y = CENTER_Y - nose_y

    if err_x >  DEADZONE_PX:
        parts.append("LEFT")   # 코가 오른쪽 → 카메라를 왼쪽으로
    elif err_x < -DEADZONE_PX:
        parts.append("RIGHT")  # 코가 왼쪽  → 카메라를 오른쪽으로

    if err_y >  DEADZONE_PX:
        parts.append("UP")     # 코가 아래쪽 → 카메라를 위로
    elif err_y < -DEADZONE_PX:
        parts.append("DOWN")   # 코가 위쪽  → 카메라를 아래로

    return "+".join(parts) if parts else "CENTER"


# ──────────────────────────────────────────────
# 화면 오버레이 그리기
# ──────────────────────────────────────────────

def draw_overlay(frame, fx, fy, fw, fh, nose_x, nose_y, direction):
    # 얼굴 박스
    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (180, 180, 0), 1)

    # 중앙 십자선
    cv2.line(frame, (CENTER_X - 30, CENTER_Y), (CENTER_X + 30, CENTER_Y), (0, 255, 0), 1)
    cv2.line(frame, (CENTER_X, CENTER_Y - 30), (CENTER_X, CENTER_Y + 30), (0, 255, 0), 1)

    # 코 위치 원
    cv2.circle(frame, (nose_x, nose_y), 8, (0, 0, 255), -1)

    # 코 → 중앙 선
    cv2.line(frame, (nose_x, nose_y), (CENTER_X, CENTER_Y), (255, 100, 0), 1)

    # 방향 텍스트
    color = (0, 255, 0) if direction == "CENTER" else (0, 100, 255)
    cv2.putText(frame, f"Direction: {direction}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # 좌표 및 오차 표시
    cv2.putText(frame, f"Nose: ({nose_x}, {nose_y})", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    err_x = CENTER_X - nose_x
    err_y = CENTER_Y - nose_y
    cv2.putText(frame, f"Err: x={err_x:+d}  y={err_y:+d}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # 방향 화살표 (중앙 → 코 방향)
    if direction != "CENTER":
        dx = nose_x - CENTER_X
        dy = nose_y - CENTER_Y
        end_x = CENTER_X + int(dx * 0.4)
        end_y = CENTER_Y + int(dy * 0.4)
        cv2.arrowedLine(frame, (CENTER_X, CENTER_Y), (end_x, end_y),
                        (0, 100, 255), 3, tipLength=0.3)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main(use_motor: bool):
    # 모터 연결 시도 (옵션)
    tracker = None
    if use_motor:
        try:
            from face_tracker_xl430_06_14 import Tracker
            tracker = Tracker()
            print("[OK] Dynamixel 모터 연결 완료")
        except Exception as e:
            print(f"[WARN] 모터 연결 실패 → DRY_RUN 으로 전환: {e}")

    # OpenCV Haar cascade 초기화 (opencv 내장 파일, 별도 다운로드 불필요)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("[ERROR] Haar cascade 파일을 로드할 수 없습니다.")
        sys.exit(1)

    # 웹캠 열기
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        sys.exit(1)

    print("[OK] 웹캠 시작 — q 키로 종료")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)   # 좌우 반전 (거울 모드)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            if len(faces) > 0:
                # 가장 큰 얼굴 하나만 사용
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                nose_x, nose_y = estimate_nose(fx, fy, fw, fh)

                direction = get_direction(nose_x, nose_y)
                draw_overlay(frame, fx, fy, fw, fh, nose_x, nose_y, direction)

                if tracker is not None:
                    tracker.update(nose_x, nose_y)
            else:
                cv2.putText(frame, "No face detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if tracker is not None:
                    tracker.update(None, None)

            cv2.imshow("Face Tracker Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if tracker is not None:
            tracker.close()
            print("[OK] 모터 토크 OFF, 포트 닫힘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="얼굴 추적 테스트")
    parser.add_argument("--motor", action="store_true",
                        help="Dynamixel 모터 제어 활성화 (기본: 화면 표시만)")
    args = parser.parse_args()
    main(use_motor=args.motor)
