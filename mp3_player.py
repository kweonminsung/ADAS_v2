#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
from typing import Optional


class MP3LoopPlayer:
    def __init__(self, mp3_path: str, enabled: bool = True):
        self.mp3_path = os.path.abspath(mp3_path)
        self.enabled = enabled
        self.proc: Optional[subprocess.Popen] = None
        self._cmd = self._select_command()
        self._warned = False

    def _select_command(self):
        if shutil.which("mpg123"):
            # -q: quiet, --loop -1: 무한 반복
            return ["mpg123", "-q", "--loop", "-1", self.mp3_path]

        if shutil.which("ffplay"):
            # -nodisp: 영상창 없음, -loop 0: 무한 반복
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-loop", "0", self.mp3_path]

        if os.name == "nt":
            powershell = shutil.which("powershell") or shutil.which("powershell.exe")
            if powershell:
                mp3_path = self.mp3_path.replace("'", "''")
                script = (
                    "Add-Type -AssemblyName PresentationCore;"
                    f"$path='{mp3_path}';"
                    "$player=New-Object System.Windows.Media.MediaPlayer;"
                    "$player.Open([System.Uri]::new($path));"
                    "$player.MediaEnded += { $player.Position = [TimeSpan]::Zero; $player.Play() };"
                    "$player.Play();"
                    "while ($true) { Start-Sleep -Milliseconds 200 }"
                )
                return [
                    powershell,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-STA",
                    "-Command",
                    script,
                ]

        return None

    def start(self):
        """이미 재생 중이면 다시 시작하지 않고, 아니면 mp3 반복 재생 시작."""
        if not self.enabled:
            return

        if self.proc is not None and self.proc.poll() is None:
            return

        if not os.path.exists(self.mp3_path):
            if not self._warned:
                print(f"[WARN] mp3 파일을 찾을 수 없습니다: {self.mp3_path}")
                self._warned = True
            return

        if self._cmd is None:
            if not self._warned:
                print("[WARN] mp3 재생기를 찾을 수 없습니다. Windows에서는 PowerShell, 또는 ffplay/mpg123 설치를 확인하세요.")
                self._warned = True
            return

        try:
            popen_kwargs = {}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            self.proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                **popen_kwargs,
            )
        except Exception as e:
            if not self._warned:
                print(f"[WARN] mp3 재생 시작 실패: {e}")
                self._warned = True
            self.proc = None

    def stop(self):
        """재생 중이면 중지."""
        if self.proc is None:
            return

        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()

        self.proc = None

    def update(self, should_play: bool):
        """should_play=True이면 재생, False이면 정지."""
        if should_play:
            self.start()
        else:
            self.stop()

    def close(self):
        self.stop()
