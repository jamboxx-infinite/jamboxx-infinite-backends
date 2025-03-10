import logging
import subprocess
from fastapi import FastAPI
from app.routers import router
import os

logger = logging.getLogger(__name__)

def verify_ffmpeg():
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
    ffprobe_path = r"C:\ffmpeg\bin\ffprobe.exe"
    
    try:
        # 验证文件存在
        if not os.path.exists(ffmpeg_path):
            raise FileNotFoundError(f"FFmpeg not found at {ffmpeg_path}")
        if not os.path.exists(ffprobe_path):
            raise FileNotFoundError(f"FFprobe not found at {ffprobe_path}")
            
        # 验证可执行性
        subprocess.run([ffmpeg_path, "-version"], check=True, capture_output=True)
        subprocess.run([ffprobe_path, "-version"], check=True, capture_output=True)
        
        # 设置环境变量
        os.environ["FFMPEG_BINARY"] = ffmpeg_path
        os.environ["FFPROBE_BINARY"] = ffprobe_path
        
        logger.info("FFmpeg verified successfully")
        return True
    except Exception as e:
        logger.error(f"FFmpeg verification failed: {str(e)}")
        return False

app = FastAPI(title="Jamboxx Backend API")

# 在应用启动时验证
@app.on_event("startup")
async def startup_event():
    if not verify_ffmpeg():
        raise RuntimeError("FFmpeg configuration failed")

# 注册路由
app.include_router(router.router)

@app.get("/ping")
def ping():
    """健康检查接口"""
    return {"status": "ok", "message": "pong"}

if __name__ == "__main__":
    import uvicorn
    """
    启动服务器的方式：
    1. 使用 uvicorn 命令（推荐）：
       uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    
    2. 直接运行此文件：
       python app/main.py
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)
