@echo off
setlocal
cd /d "%~dp0"
python -m pip install -r requirements.txt pyinstaller -q
python -m PyInstaller --noconfirm --clean ^
  --onefile --windowed ^
  --name KDCJC ^
  --collect-all cryptography ^
  main.py
echo.
echo 输出: dist\KDCJC.exe
pause
