#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모터 단독 테스트 — 웹캠/얼굴인식 없음

키보드로 가상 코 좌표를 직접 입력해 Tracker.update() 를 호출합니다.
모터 없이 실행하면 어떤 좌표가 전송될지만 출력합니다.

실행:
  python3 motor_test.py            # DRY_RUN (좌표 출력만)
  python3 motor_test.py --motor    # 실제 모터 제어
"""

import argparse
import time

# ──────────────────────────────────────────────
# 화면 해상도 기준 (기존 코드와 동일하게 맞춤)
# ──────────────────────────────────────────────
FRAME_W, FRAME_H   = 1280, 720
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2   # 640, 360

# 방향키 한 번 누를 때 이동할 픽셀
STEP = 50

# ──────────────────────────────────────────────
# 미리 정의된 테스트 위치
# ──────────────────────────────────────────────
PRESETS = {
    "c": ("CENTER",     CENTER_X,          CENTER_Y),
    "l": ("LEFT",       CENTER_X - 200,    CENTER_Y),
    "r": ("RIGHT",      CENTER_X + 200,    CENTER_Y),
    "u": ("UP",         CENTER_X,          CENTER_Y - 200),
    "d": ("DOWN",       CENTER_X,          CENTER_Y + 200),
    "q": ("QUIT",       None,              None),
}

MENU = """
──────────────────────────────────────
  c       : CENTER  (640, 360)
  l       : LEFT    (440, 360)
  r       : RIGHT   (840, 360)
  u       : UP      (640, 160)
  d       : DOWN    (640, 560)
  x y     : 좌표 직접 입력  예) 400 200
  a       : 자동 순환 테스트
  q       : 종료
──────────────────────────────────────"""


def send(tracker, label, x, y):
    """좌표 출력 후 tracker 에 전송 (tracker=None 이면 dry-run)."""
    err_x = CENTER_X - x if x is not None else 0
    err_y = CENTER_Y - y if y is not None else 0
    print(f"  [{label}]  nose=({x}, {y})  err=(x:{err_x:+d}, y:{err_y:+d})")
    if tracker is not None:
        tracker.update(x, y)


def auto_cycle(tracker):
    """CENTER → LEFT → RIGHT → UP → DOWN → CENTER 순서로 자동 순환."""
    sequence = [
        ("CENTER", CENTER_X,       CENTER_Y),
        ("LEFT",   CENTER_X - 200, CENTER_Y),
        ("CENTER", CENTER_X,       CENTER_Y),
        ("RIGHT",  CENTER_X + 200, CENTER_Y),
        ("CENTER", CENTER_X,       CENTER_Y),
        ("UP",     CENTER_X,       CENTER_Y - 200),
        ("CENTER", CENTER_X,       CENTER_Y),
        ("DOWN",   CENTER_X,       CENTER_Y + 200),
        ("CENTER", CENTER_X,       CENTER_Y),
    ]
    print("\n  [자동 순환 시작] 각 방향 1.5초 유지 — Ctrl+C 로 중단\n")
    try:
        for label, x, y in sequence:
            send(tracker, label, x, y)
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\n  [자동 순환 중단]")


def main(use_motor: bool):
    tracker = None
    if use_motor:
        try:
            from face_tracker_xl430_06_14 import Tracker
            tracker = Tracker()
            print("[OK] Dynamixel 모터 연결 완료")
        except Exception as e:
            print(f"[WARN] 모터 연결 실패 → DRY_RUN 으로 전환: {e}")
    else:
        print("[DRY_RUN] 좌표 출력만 합니다 (모터 없음)")

    print(MENU)

    while True:
        try:
            key = input("명령 입력 > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        parts = key.split()
        if key == "q":
            break
        elif key == "a":
            auto_cycle(tracker)
        elif len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
            x, y = int(parts[0]), int(parts[1])
            send(tracker, "CUSTOM", x, y)
        elif key in PRESETS:
            label, x, y = PRESETS[key]
            send(tracker, label, x, y)
        else:
            print("  [?] 알 수 없는 명령. 위 메뉴를 참고하세요.")

    if tracker is not None:
        tracker.close()
        print("[OK] 모터 토크 OFF, 포트 닫힘")
    print("종료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모터 단독 테스트 (웹캠 없음)")
    parser.add_argument("--motor", action="store_true",
                        help="실제 Dynamixel 모터 제어 활성화")
    args = parser.parse_args()
    main(use_motor=args.motor)
