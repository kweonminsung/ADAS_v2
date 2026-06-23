#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — oCam + YOLOv8 pose + Dynamixel 통합

동작 흐름:
  1. oCam(video4)으로 영상 수신
  2. YOLOv8 pose로 사람 감지 + 관절 추출
  3. 손을 3초 이상 든 사람을 트래킹 대상으로 확정
  4. 확정된 사람의 코 좌표 → Tracker.update() → 모터 제어

실행:
  python3 main.py           # 모터 없이 화면만
  python3 main.py --motor   # 모터 제어 포함
"""

import argparse
import os
import sys
import time
from collections import defaultdict

# oCam 비표준 포맷 처리를 위해 libv4l2 자동 적용
_V4L2_LIB = "/usr/lib/x86_64-linux-gnu/libv4l/v4l2convert.so"
if os.path.exists(_V4L2_LIB) and "LD_PRELOAD" not in os.environ:
    os.environ["LD_PRELOAD"] = _V4L2_LIB
    os.execv(sys.executable, [sys.executable] + sys.argv)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import cv2
from ultralytics import YOLO
from mp3_player import MP3LoopPlayer

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
CAMERA_IDX   = 0                     # oCam 카메라 인덱스
MODEL_PATH   = "yolov8n-pose.pt"     # YOLO pose 모델

# 트래커가 기준으로 쓰는 해상도 (face_tracker_xl430_06_14.py 와 동일)
TRACKER_W, TRACKER_H = 1280, 720
ROUTE_EXIT_MP3 = os.path.join(BASE_DIR, "route_exit.mp3")

# 손 든 상태를 몇 초 유지해야 트래킹 확정할지
ARM_UP_THRESHOLD = 3.0               # 초

# 손이 내려간 것으로 판정하기까지 허용할 연속 프레임 수
# (YOLO 키포인트 노이즈로 인한 순간 끊김 방지)
ARM_DOWN_GRACE = 10                  # 프레임

# 얼굴 박스 패딩 (헤드 keypoint 범위 밖으로 여유 픽셀)
FACE_BOX_PAD = 30                    # 픽셀

# YOLOv8 pose keypoint 인덱스 (COCO 기준)
KP_NOSE          = 0
KP_LEFT_EYE      = 1
KP_RIGHT_EYE     = 2
KP_LEFT_EAR      = 3
KP_RIGHT_EAR     = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER= 6
KP_LEFT_WRIST    = 9
KP_RIGHT_WRIST   = 10


# ──────────────────────────────────────────────
# 얼굴 박스 추정 (헤드 keypoint 0~4 기반)
# ──────────────────────────────────────────────

def estimate_face_box(kp, pad=FACE_BOX_PAD):
    """코·눈·귀 keypoint(0~4)로 얼굴 영역 박스를 추정."""
    head_kp = kp[:5]  # nose, left_eye, right_eye, left_ear, right_ear
    xs = [p[0] for p in head_kp if p[0] > 0]
    ys = [p[1] for p in head_kp if p[1] > 0]
    if not xs or not ys:
        return None
    return (int(min(xs) - pad), int(min(ys) - pad),
            int(max(xs) + pad), int(max(ys) + pad))

def in_face_box(x, y, box):
    """좌표가 얼굴 박스 안에 있으면 True."""
    if box is None:
        return True  # 박스 없으면 통과
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


# ──────────────────────────────────────────────
# 손 들었는지 판단
# ──────────────────────────────────────────────

def is_arm_up(kp):
    """keypoints 배열(N×2)에서 손목이 어깨보다 위에 있으면 True."""
    if len(kp) <= KP_RIGHT_WRIST:
        return False
    left_up  = kp[KP_LEFT_WRIST][1]  < kp[KP_LEFT_SHOULDER][1]
    right_up = kp[KP_RIGHT_WRIST][1] < kp[KP_RIGHT_SHOULDER][1]
    return left_up or right_up


# ──────────────────────────────────────────────
# 화면 오버레이
# ──────────────────────────────────────────────

def draw_info(frame, nose_x, nose_y, arm_up, confirmed, elapsed, box, fw, fh):
    cx, cy = fw // 2, fh // 2

    # 중앙 십자선
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 1)

    # ── 상태 1: 아무도 없음 ──
    if not arm_up and not confirmed:
        cv2.putText(frame, "No target", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return

    # ── 상태 2: 손 올림 대기 중 ──
    if arm_up and not confirmed:
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "ARM UP!", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "ARM UP! Holding...", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        bar_x, bar_y, bar_w, bar_h = 20, 75, 300, 20
        ratio = min(elapsed / ARM_UP_THRESHOLD, 1.0)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * ratio), bar_y + bar_h), (0, 255, 255), -1)
        cv2.putText(frame, f"{elapsed:.1f}s / {ARM_UP_THRESHOLD:.0f}s", (bar_x + bar_w + 10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        return

    # ── 상태 3: 트래킹 확정 ──
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "TARGET", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if nose_x is not None:
        cv2.circle(frame, (nose_x, nose_y), 7, (0, 0, 255), -1)
        cv2.line(frame, (nose_x, nose_y), (cx, cy), (255, 100, 0), 1)
        cv2.putText(frame, f"TRACKING  Nose:({nose_x},{nose_y})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main(use_motor: bool):
    route_exit_player = MP3LoopPlayer(ROUTE_EXIT_MP3)

    # 모터 초기화
    tracker = None
    if use_motor:
        try:
            from face_tracker_xl430_06_14 import Tracker
            tracker = Tracker()
            print("[OK] Dynamixel 모터 연결 완료")
        except Exception as e:
            print(f"[WARN] 모터 연결 실패 → DRY_RUN: {e}")

    # YOLO 모델 로드
    print("[..] YOLO 모델 로딩 중...")
    model = YOLO(MODEL_PATH)
    print("[OK] YOLO 로드 완료")

    # 카메라 열기
    cap = cv2.VideoCapture(CAMERA_IDX)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        sys.exit(1)

    FRAME_W, FRAME_H = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # 카메라 워밍업 (초반 검은 프레임 버리기)
    print("[..] 카메라 워밍업 중...")
    for _ in range(20):
        cap.read()
    print(f"[OK] 카메라 시작: {FRAME_W}x{FRAME_H}  — q 키로 종료")

    # 트래킹 상태
    arm_up_start   = defaultdict(lambda: None)
    arm_confirmed  = defaultdict(lambda: False)
    arm_down_count = defaultdict(lambda: 0)
    tracked_id     = None
    last_nose_x, last_nose_y = None, None
    last_face_box  = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            results = model(frame, verbose=False)

            nose_x, nose_y = None, None
            elapsed = 0.0
            current_face_box = None

            for r in results:
                if r.keypoints is None or r.boxes is None:
                    continue

                kp_all  = r.keypoints.xy          # (N, 17, 2)
                boxes   = r.boxes.xyxy            # (N, 4)
                n_person = len(kp_all)

                # 유효하지 않은 tracked_id 초기화
                if tracked_id is not None and tracked_id >= n_person:
                    tracked_id = None

                # 트래킹 대상 없으면 손 든 사람 중 가장 큰 박스 선택
                if tracked_id is None:
                    best_area, best_id = 0, None
                    for i in range(n_person):
                        kp = kp_all[i].cpu().numpy()
                        if not is_arm_up(kp):
                            continue
                        x1, y1, x2, y2 = boxes[i].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        if area > best_area:
                            best_area, best_id = area, i
                    if best_id is not None:
                        tracked_id = best_id

                # 트래킹 대상 처리
                if tracked_id is not None:
                    kp = kp_all[tracked_id].cpu().numpy()
                    arm_now = is_arm_up(kp)
                    current_face_box = estimate_face_box(kp)  # 대기 중 표시용

                    if arm_now:
                        arm_down_count[tracked_id] = 0
                        if arm_up_start[tracked_id] is None:
                            arm_up_start[tracked_id] = time.time()
                        elapsed = time.time() - arm_up_start[tracked_id]
                        if elapsed >= ARM_UP_THRESHOLD:
                            arm_confirmed[tracked_id] = True
                    else:
                        arm_down_count[tracked_id] += 1
                        # grace period 내에는 트래킹 유지, 초과 시에만 해제
                        if arm_down_count[tracked_id] > ARM_DOWN_GRACE:
                            arm_up_start[tracked_id]  = None
                            arm_confirmed[tracked_id] = False
                            arm_down_count[tracked_id] = 0
                            tracked_id = None
                            last_nose_x, last_nose_y = None, None
                            last_face_box = None

                    if tracked_id is not None and arm_confirmed[tracked_id]:
                        raw_x = int(kp[KP_NOSE][0])
                        raw_y = int(kp[KP_NOSE][1])

                        # 얼굴 박스 필터: 이전 얼굴 박스 안에 있는 좌표만 수락
                        if last_nose_x is None:
                            # 첫 확정 → 그대로 사용하고 얼굴 박스 초기화
                            last_nose_x, last_nose_y = raw_x, raw_y
                            last_face_box = estimate_face_box(kp)
                        elif in_face_box(raw_x, raw_y, last_face_box):
                            # 얼굴 박스 안 → 수락 후 박스 갱신
                            last_nose_x, last_nose_y = raw_x, raw_y
                            last_face_box = estimate_face_box(kp)
                        # 얼굴 박스 밖 → 무시, 이전 좌표 유지

                        # 카메라 해상도 → 트래커 기준 해상도로 스케일
                        nose_x = int(last_nose_x * TRACKER_W / FRAME_W)
                        nose_y = int(last_nose_y * TRACKER_H / FRAME_H)

                        if tracker is not None:
                            tracker.update(nose_x, nose_y)

                        # 화면에는 실제 픽셀 좌표로 표시
                        nose_x, nose_y = last_nose_x, last_nose_y

            is_tracking   = tracked_id is not None and arm_confirmed[tracked_id]
            is_arm_raised = tracked_id is not None and not is_tracking
            display_box   = last_face_box if is_tracking else current_face_box
            route_exit_player.update(is_tracking)
            draw_info(frame, nose_x, nose_y, is_arm_raised, is_tracking, elapsed, display_box, FRAME_W, FRAME_H)
            cv2.imshow("Face Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        route_exit_player.close()
        cap.release()
        cv2.destroyAllWindows()
        if tracker is not None:
            tracker.close()
            print("[OK] 모터 토크 OFF, 포트 닫힘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motor", action="store_true", help="Dynamixel 모터 제어 활성화")
    args = parser.parse_args()
    main(use_motor=args.motor)
