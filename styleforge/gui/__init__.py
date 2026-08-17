"""StyleForge 데스크톱 GUI (tkinter).

CLI(`styleforge/cli.py`)와 나란히 있는 또 하나의 얇은 진입점이다. 로직은
전부 기존 `train/apply/sweep`의 runner 함수에 있고, 이 패키지는 폼 입력을
그 함수의 키워드 인자로 옮기고 진행률 콜백을 위젯에 반영하는 와이어링만
한다 (CLAUDE.md 8장 "CLI 계층은 얇게 유지" 원칙을 GUI에도 그대로 적용).
"""
