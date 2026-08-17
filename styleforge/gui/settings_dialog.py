"""ComfyUI 경로 설정 다이얼로그.

`.env`의 COMFY_START_CWD/COMFY_LORA_DIR을 사용자가 직접 텍스트 에디터로
열지 않고 GUI에서 바로 찾아보기로 고칠 수 있게 한다. 경로를 하드코딩하지
않고 .env -> config.py로 주입하는 원칙(CLAUDE.md 3장)은 그대로 유지한다 —
이 다이얼로그는 그 .env 파일을 사용자 대신 갱신해주는 창구일 뿐이다.
외장 드라이브 미연결 등으로 ComfyUI 경로가 컴퓨터마다 달라 자주 깨지는
문제(WinError 267)를 GUI에서 바로 고칠 수 있게 하기 위해 추가됐다.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from styleforge.config import settings, update_env_file


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.title("설정")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        form = ttk.Frame(self, padding=10)
        form.pack(fill="both", expand=True)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="ComfyUI 실행 파일 폴더").grid(row=0, column=0, sticky="w", pady=3)
        self.cwd_var = tk.StringVar(value=str(settings.comfy_start_cwd) if settings.comfy_start_cwd else "")
        ttk.Entry(form, textvariable=self.cwd_var, width=48).grid(row=0, column=1, sticky="ew", padx=(6, 4))
        ttk.Button(form, text="찾아보기", command=lambda: self._browse(self.cwd_var)).grid(row=0, column=2)

        ttk.Label(form, text="ComfyUI LoRA 폴더").grid(row=1, column=0, sticky="w", pady=3)
        self.lora_var = tk.StringVar(value=str(settings.comfy_lora_dir) if settings.comfy_lora_dir else "")
        ttk.Entry(form, textvariable=self.lora_var, width=48).grid(row=1, column=1, sticky="ew", padx=(6, 4))
        ttk.Button(form, text="찾아보기", command=lambda: self._browse(self.lora_var)).grid(row=1, column=2)

        button_row = ttk.Frame(form)
        button_row.grid(row=2, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(button_row, text="취소", command=self.destroy).pack(side="right", padx=(4, 0))
        ttk.Button(button_row, text="저장", command=self._save).pack(side="right", padx=(0, 4))

    def _browse(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(initialdir=var.get() or None)
        if path:
            var.set(path)

    def _save(self) -> None:
        cwd = self.cwd_var.get().strip()
        lora_dir = self.lora_var.get().strip()

        if cwd and not Path(cwd).is_dir():
            messagebox.showerror("설정 오류", f"ComfyUI 실행 파일 폴더가 존재하지 않습니다:\n{cwd}")
            return
        if lora_dir and not Path(lora_dir).is_dir():
            messagebox.showerror("설정 오류", f"ComfyUI LoRA 폴더가 존재하지 않습니다:\n{lora_dir}")
            return

        update_env_file({"COMFY_START_CWD": cwd, "COMFY_LORA_DIR": lora_dir})
        # 재시작 없이 바로 반영되도록 실행 중인 settings 싱글턴도 함께 갱신한다.
        settings.comfy_start_cwd = Path(cwd) if cwd else None
        settings.comfy_lora_dir = Path(lora_dir) if lora_dir else None

        messagebox.showinfo("설정 저장됨", ".env에 저장했습니다.")
        self.destroy()
