"""StyleForge GUI 진입점. `styleforge-gui`로 실행한다 (pyproject.toml)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from styleforge.gui.apply_tab import ApplyTab
from styleforge.gui.common import ComfyStatusBar
from styleforge.gui.settings_dialog import SettingsDialog
from styleforge.gui.sweep_tab import SweepTab
from styleforge.gui.train_tab import TrainTab


class StyleForgeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("StyleForge")
        root.geometry("820x640")

        # 세 탭 중 하나라도 실행 중이면 나머지 실행 버튼도 잠근다 — 학습과
        # ComfyUI는 동시에 못 돌리고(CLAUDE.md 3장), apply/sweep도 ComfyUI를
        # 하나씩만 써야 하므로 GUI에서 여러 작업을 동시에 못 누르게 막는다.
        self._tabs: list[TrainTab | ApplyTab | SweepTab] = []

        top_row = ttk.Frame(root)
        top_row.pack(fill="x", padx=8, pady=(8, 0))

        self.comfy_bar = ComfyStatusBar(top_row, on_busy_change=self._set_busy)
        self.comfy_bar.pack(side="left")
        ttk.Button(top_row, text="설정", command=self._open_settings).pack(side="right")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        train_tab = TrainTab(notebook, on_busy_change=self._set_busy)
        apply_tab = ApplyTab(notebook, on_busy_change=self._set_busy)
        sweep_tab = SweepTab(notebook, on_busy_change=self._set_busy)
        self._tabs = [train_tab, apply_tab, sweep_tab]

        notebook.add(train_tab, text="학습 (train)")
        notebook.add(apply_tab, text="적용 (apply)")
        notebook.add(sweep_tab, text="탐색 (sweep)")

    def _set_busy(self, busy: bool) -> None:
        for tab in self._tabs:
            tab.set_enabled(not busy)
        self.comfy_bar.notify_busy(busy)

    def _open_settings(self) -> None:
        SettingsDialog(self.root)


def main() -> None:
    root = tk.Tk()
    StyleForgeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
