@echo off
echo Starting build process...

REM 设置环境变量
set PYTHONPATH=%CD%

REM 创建编译输出目录
if not exist "dist" mkdir dist

REM 执行Nuitka编译
python -m nuitka ^
    --follow-imports ^
    --include-package=app ^
    --include-package=ddsp ^
    --include-package=encoder ^
    --include-package=fairseq ^
    --include-package=torch ^
    --include-package=torchaudio ^
    --include-package=numpy ^
    --include-package=librosa ^
    --include-package=fastapi ^
    --include-package=uvicorn ^
    --include-data-dir=app/config=config ^
    --include-data-dir=app/models=models ^
    --enable-plugin=numpy ^
    --enable-plugin=torch ^
    --standalone ^
    --assume-yes-for-downloads ^
    --windows-icon-from-ico=app/assets/icon.ico ^
    --output-dir=dist ^
    app/main.py

REM 复制配置文件和模型
xcopy /E /I "app\config" "dist\main.dist\config"
xcopy /E /I "app\models" "dist\main.dist\models"

echo Build complete!