#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XL430-W250-T 2축(Yaw-Pitch) 얼굴(코) 추적 - 모터 제어 모듈

역할 분담
  - 카메라/코 검출(좌표 계산)은 "외부 코드"가 담당
  - 이 파일은 "코 좌표 (x, y) -> 모터 명령" 부분만 담당
  - 사용법: tracker.update(nose_x, nose_y) 를 매 프레임 호출

한 사이클 동작
  1) GroupSyncRead  로 두 모터의 현재 위치(Present Position, 132번 주소)를 읽음
  2) 화면 중심(640, 360) 기준 픽셀 오차 계산
  3) 새 목표 = 현재 위치 + (픽셀오차 × 환산계수 × 부호)  <- 절대 목표 펄스
  4) GroupSyncWrite 로 두 모터의 Goal Position(116번 주소)에 한 패킷으로 기록
  5) 나머지(부드러운 이동)는 모터 내장 PID 가 처리

준비물
  - Dynamixel SDK  : pip install dynamixel-sdk
  - Dynamixel Wizard 2.0 으로 Yaw=ID 1, Pitch=ID 2 미리 설정 (기본 ID는 모두 1)
  - 모터 전원(배터리/SMPS 12V) 연결, U2D2 USB 연결
"""

import time
from dynamixel_sdk import (
    PortHandler,       # 시리얼 포트 열기/닫기 담당
    PacketHandler,     # 다이나믹셀 프로토콜 2.0 패킷 생성/파싱 담당
    GroupSyncWrite,    # 여러 모터에 동시에 쓰는 Sync Write 담당
    GroupSyncRead,     # 여러 모터에서 동시에 읽는 Sync Read 담당
    DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD,  # 정수 → 바이트 분해 유틸
    COMM_SUCCESS,      # 통신 성공 반환 코드 (== 0)
)

# ============================================================
# 1) 연결 설정
# ============================================================

# U2D2 가 잡히는 시리얼 포트 이름
#   Linux  : "/dev/ttyUSB0"  (ls /dev/ttyUSB* 로 확인)
#   Mac    : "/dev/tty.usbserial-XXXX"
#   Windows: "COM3" 형태 (장치관리자에서 확인)
DEVICENAME   = "/dev/ttyUSB0"

# 다이나믹셀 통신 속도. XL430 공장 기본값은 57600 bps.
# Wizard 2.0 에서 변경했다면 여기도 맞춰야 함.
BAUDRATE     = 1000000

# XL430(X 시리즈) 은 프로토콜 2.0 사용. AX/MX 구형 모터는 1.0.
PROTOCOL_VER = 2.0

# 모터 ID. Wizard 2.0 으로 미리 설정해둔 값과 일치해야 함.
DXL_ID_YAW   = 1   # 좌우(수평) 회전 담당
DXL_ID_PITCH = 2   # 상하(수직) 회전 담당
DXL_IDS      = [DXL_ID_YAW, DXL_ID_PITCH]  # 루프용 리스트

# ============================================================
# 2) XL430 컨트롤 테이블 주소 및 크기
#    (EEPROM 영역: 0~63번, RAM 영역: 64번~)
# ============================================================

ADDR_OPERATING_MODE   = 11   # [EEPROM] 제어 모드 선택. 토크 OFF 상태에서만 변경 가능.
ADDR_TORQUE_ENABLE    = 64   # [RAM]    1=토크 ON(모터 잠금), 0=토크 OFF(수동 회전 가능)
ADDR_PROFILE_VELOCITY = 112  # [RAM]    목표 위치로 이동할 때의 최대 속도 제한 (0=무제한)
ADDR_GOAL_POSITION    = 116  # [RAM]    목표 위치(펄스 단위). 여기에 값을 쓰면 모터가 이동함.
ADDR_PRESENT_POSITION = 132  # [RAM]    현재 위치(펄스 단위). 읽기 전용.
LEN_POSITION          = 4    # 위치 값은 4바이트(32비트) 정수

# Operating Mode 값 정의
OP_MODE_POSITION = 3   # 위치 제어 모드: 0~4095 펄스 범위 (1회전 = 4096 펄스)

# Torque Enable 값 정의
TORQUE_ENABLE  = 1
TORQUE_DISABLE = 0

# 안전 가동 범위(펄스 단위)
#   4096 펄스 = 360도 → 1 펄스 ≈ 0.088도
#   2048 = 정면(180도), 1024 = -90도, 3072 = +90도
POS_MIN    = 1024   # 최소 위치 (정면 기준 -90도)
POS_MAX    = 3072   # 최대 위치 (정면 기준 +90도)
POS_CENTER = 2048   # 정면(중앙) 위치

# ============================================================
# 3) 화면 / 추적 파라미터 (튜닝 대상)
# ============================================================

FRAME_W, FRAME_H   = 1280, 720
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2   # 추적 기준점 (640, 360)

# 픽셀 오차 → 펄스 이동량 환산계수
#   이론값(수평 화각 65도 기준): (65 / 1280) × (4096 / 360) ≈ 0.578
#   실제로는 렌즈 왜곡·크롭 등으로 오버슈트가 발생하므로 보수적으로 시작.
#   튜닝 방법: 굼뜨면 키우고(0.3→0.4), 진동하면 줄이기(0.2→0.1)
CONVERSION = 0.20

# 모터 장착 방향에 따라 부호를 반전해야 할 수 있음.
#   얼굴이 오른쪽으로 가는데 카메라가 왼쪽으로 돌면 -1 로 바꿀 것.
SIGN_YAW   = -1   # 화면 x 오차 → yaw(좌우) 모터 방향
SIGN_PITCH = -1   # 화면 y 오차 → pitch(상하) 모터 방향

# 데드존: 이 픽셀 수 이내의 오차는 0으로 처리.
# 너무 작으면 모터가 미세하게 계속 떨림(헌팅), 너무 크면 추적이 부정확해짐.
DEADZONE_PX = 20

# 한 프레임에 모터가 이동할 수 있는 최대 펄스.
# 얼굴 검출이 순간적으로 튀었을 때 모터가 급격히 움직이는 것을 방지.
MAX_STEP = 500

# 모터 내부 속도 프로파일 제한값 (단위: rev/min 환산 내부 단위).
# 0 = 최고속(동작이 거칠어짐), 100 정도면 부드럽게 이동.
PROFILE_VELOCITY = 100


# ============================================================
# 유틸 함수
# ============================================================

def _clamp(v, lo, hi):
    """v 를 [lo, hi] 범위로 제한."""
    return max(lo, min(hi, v))


def _to_4byte(value):
    """
    정수값을 SyncWrite 에 넣을 수 있는 4바이트 리스트(리틀엔디안)로 변환.
    예) 2048(0x00000800) → [0x00, 0x08, 0x00, 0x00]
    """
    return [
        DXL_LOBYTE(DXL_LOWORD(value)),   # 바이트 0: 하위 워드의 하위 바이트 (LSB)
        DXL_HIBYTE(DXL_LOWORD(value)),   # 바이트 1: 하위 워드의 상위 바이트
        DXL_LOBYTE(DXL_HIWORD(value)),   # 바이트 2: 상위 워드의 하위 바이트
        DXL_HIBYTE(DXL_HIWORD(value)),   # 바이트 3: 상위 워드의 상위 바이트 (MSB)
    ]


# ============================================================
# Tracker 클래스
# ============================================================

class Tracker:
    def __init__(self):
        # PortHandler: 시리얼 포트(U2D2)를 열고 닫는 역할
        self.port = PortHandler(DEVICENAME)
        # PacketHandler: 프로토콜 2.0 기반으로 패킷을 만들고 해석하는 역할
        self.packet = PacketHandler(PROTOCOL_VER)

        # 포트 열기 실패 시 즉시 종료 (드라이버 미설치, 포트 이름 오류, 권한 부족 등)
        if not self.port.openPort():
            raise IOError("포트 열기 실패 — DEVICENAME 또는 USB 연결 확인")
        if not self.port.setBaudRate(BAUDRATE):
            raise IOError("보율 설정 실패 — BAUDRATE 값 확인")

        # --- 두 모터 초기화 (순서 중요) ---
        for dxl_id in DXL_IDS:
            # ① 토크 OFF: EEPROM 영역(Operating Mode)은 토크가 꺼진 상태에서만 쓸 수 있음
            self._w1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

            # ② 위치 제어 모드(3번)로 설정
            #    이 설정은 전원이 꺼져도 EEPROM에 저장됨
            self._w1(dxl_id, ADDR_OPERATING_MODE, OP_MODE_POSITION)

            # ③ 속도 프로파일 설정: 목표 위치로 이동할 때의 최대 속도 제한
            #    토크 ON 전에 설정해야 첫 이동부터 적용됨
            self._w4(dxl_id, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY)

            # ④ 토크 ON: 이 시점부터 모터가 위치를 유지하려 힘을 가함
            self._w1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

        # --- GroupSyncWrite 초기화 ---
        # 116번 주소(Goal Position)에 4바이트를 여러 모터에 동시 전송하는 객체
        self.sync_write = GroupSyncWrite(
            self.port, self.packet, ADDR_GOAL_POSITION, LEN_POSITION)

        # --- GroupSyncRead 초기화 ---
        # 132번 주소(Present Position)에서 4바이트를 여러 모터에서 동시 수신하는 객체
        self.sync_read = GroupSyncRead(
            self.port, self.packet, ADDR_PRESENT_POSITION, LEN_POSITION)

        # SyncRead 에 읽을 모터 ID를 미리 등록 (한 번만 하면 됨)
        for dxl_id in DXL_IDS:
            if not self.sync_read.addParam(dxl_id):
                raise RuntimeError(f"SyncRead addParam 실패 id={dxl_id}")

        # 초기화 완료 후 두 모터를 정면(2048 펄스)으로 이동
        self._write_goals(POS_CENTER, POS_CENTER)
        time.sleep(0.5)   # 모터가 정면에 도달할 때까지 대기

    # --------------------------------------------------------
    # 저수준 래퍼: 단일 모터에 값을 개별 전송 (초기화 시에만 사용)
    # --------------------------------------------------------

    def _w1(self, dxl_id, addr, val):
        """1바이트 값을 특정 모터의 특정 주소에 쓰고 응답을 기다림 (TxRx)."""
        self.packet.write1ByteTxRx(self.port, dxl_id, addr, val)

    def _w4(self, dxl_id, addr, val):
        """4바이트 값을 특정 모터의 특정 주소에 쓰고 응답을 기다림 (TxRx)."""
        self.packet.write4ByteTxRx(self.port, dxl_id, addr, val)

    # --------------------------------------------------------
    # 두 모터 현재 위치를 GroupSyncRead 로 한 번에 읽기
    # --------------------------------------------------------

    def _read_present(self):
        """
        두 모터의 현재 위치(펄스)를 {id: 펄스} 딕셔너리로 반환.
        통신 실패나 모터 응답 없으면 None 반환 → 호출부에서 해당 프레임 스킵.
        """
        # txRxPacket(): "ID1, ID2 — 132번 주소에서 4바이트씩 줘" 요청을 전송하고
        # 두 모터의 응답 패킷을 내부 버퍼에 저장
        if self.sync_read.txRxPacket() != COMM_SUCCESS:
            return None   # 버스 오류, 케이블 불량 등

        out = {}
        for dxl_id in DXL_IDS:
            # isAvailable(): 해당 모터의 데이터가 버퍼에 정상적으로 들어왔는지 확인
            if not self.sync_read.isAvailable(dxl_id, ADDR_PRESENT_POSITION,
                                              LEN_POSITION):
                return None   # 특정 모터가 응답하지 않은 경우

            # getData(): 버퍼에서 값을 꺼냄 (이 시점은 통신 없이 메모리 읽기만)
            out[dxl_id] = self.sync_read.getData(dxl_id, ADDR_PRESENT_POSITION,
                                                 LEN_POSITION)
        return out   # 예: {1: 2050, 2: 2045}

    # --------------------------------------------------------
    # 두 모터 목표 위치를 GroupSyncWrite 로 한 번에 전송
    # --------------------------------------------------------

    def _write_goals(self, goal_yaw, goal_pitch):
        """
        두 모터의 Goal Position(116번)에 목표 펄스를 한 패킷으로 동시 전송.
        SyncWrite는 응답 패킷이 없어(Tx Only) 매우 빠름.
        """
        # clearParam(): 이전 프레임에서 addParam 으로 쌓인 데이터를 초기화
        # 비우지 않으면 같은 ID 가 중복 등록되어 패킷이 오염됨
        self.sync_write.clearParam()

        # addParam(): "이 ID 의 모터에게 이 바이트열을 쓸 것"을 내부 큐에 등록
        # _to_4byte() 로 정수를 리틀엔디안 4바이트로 변환 후 bytes 로 포장
        self.sync_write.addParam(DXL_ID_YAW,   bytes(_to_4byte(goal_yaw)))
        self.sync_write.addParam(DXL_ID_PITCH, bytes(_to_4byte(goal_pitch)))

        # txPacket(): 등록된 내용을 하나의 SyncWrite 패킷으로 묶어 U2D2 로 전송
        # 두 모터가 같은 패킷을 받으므로 동기화된 시점에 이동을 시작함
        self.sync_write.txPacket()

    # --------------------------------------------------------
    # 핵심 제어 루프: 매 프레임 외부에서 호출
    # --------------------------------------------------------

    def update(self, nose_x, nose_y):
        """
        외부 검출기에서 받은 코 픽셀 좌표로 한 사이클 제어.
        코가 검출되지 않았으면 (None, None) 을 전달하면 됨 → 모터 정지 유지.
        """
        # 코가 없으면(검출 실패) 이번 프레임은 스킵, 이전 목표 유지
        if nose_x is None or nose_y is None:
            return

        # 두 모터의 현재 위치를 읽음. 실패 시 이번 프레임 스킵.
        present = self._read_present()
        if present is None:
            return

        # --- 픽셀 오차 계산 ---
        # 양수 err_x: 코가 중심보다 왼쪽 → 카메라를 왼쪽으로 돌려야 함
        # 양수 err_y: 코가 중심보다 위쪽   → 카메라를 위쪽으로 돌려야 함
        err_x = CENTER_X - nose_x   # 수평 오차 (yaw 제어)
        err_y = CENTER_Y - nose_y   # 수직 오차 (pitch 제어)

        # --- 데드존 적용 ---
        # DEADZONE_PX 이내의 미세 오차는 0으로 처리하여 모터 떨림(헌팅) 방지
        if abs(err_x) < DEADZONE_PX:
            err_x = 0
        if abs(err_y) < DEADZONE_PX:
            err_y = 0

        # 두 축 모두 데드존 안에 있으면 아무것도 하지 않음
        if err_x == 0 and err_y == 0:
            return

        # --- 픽셀 오차 → 펄스 이동량 변환 ---
        # CONVERSION: 픽셀 1개당 몇 펄스 움직일지 결정하는 비례 게인
        # SIGN: 모터 장착 방향에 따라 부호 조정
        step_yaw   = SIGN_YAW   * err_x * CONVERSION
        step_pitch = SIGN_PITCH * err_y * CONVERSION

        # --- 프레임당 최대 이동량 제한 ---
        # 얼굴 검출이 순간적으로 크게 튀었을 때 모터가 급격히 움직이는 것을 방지
        step_yaw   = _clamp(step_yaw,   -MAX_STEP, MAX_STEP)
        step_pitch = _clamp(step_pitch, -MAX_STEP, MAX_STEP)

        # --- 새 절대 목표 위치 계산 ---
        # 현재 위치에 이동량을 더하고, 가동 범위(1024~3072)를 벗어나지 않도록 클램프
        goal_yaw   = _clamp(int(present[DXL_ID_YAW]   + step_yaw),   POS_MIN, POS_MAX)
        goal_pitch = _clamp(int(present[DXL_ID_PITCH] + step_pitch), POS_MIN, POS_MAX)

        # 계산된 목표 위치를 모터로 전송
        self._write_goals(goal_yaw, goal_pitch)

    def close(self):
        """종료 시 두 모터의 토크를 OFF. 전원 차단 전에 반드시 호출."""
        for dxl_id in DXL_IDS:
            self._w1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self.port.closePort()


# ============================================================
# 사용 예시 (외부 검출 루프와 연결하는 부분)
# ============================================================
# def get_nose_from_your_detector():
#     """
#     당신의 카메라/코 검출 코드로 교체.
#     코가 보이면 (x, y) 픽셀 좌표, 안 보이면 (None, None) 반환.
#     """
#     raise NotImplementedError("여기에 코 좌표 검출 코드를 연결하세요")


if __name__ == "__main__":
    tracker = Tracker()
    try:
        while True:
            x, y = get_nose_from_your_detector()   # 외부 검출 함수로 교체
            tracker.update(x, y)
            time.sleep(0.02)   # 약 50Hz (20ms 간격)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
