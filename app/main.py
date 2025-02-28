import os
import sys
from fastapi import FastAPI
from app.routers import router

# 添加资源文件路径处理
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

app = FastAPI(title="Jamboxx Backend API")
app.include_router(router.router)

@app.get("/ping")
def ping():
    """健康检查接口"""
    return {"status": "ok", "message": "pong"}

if __name__ == "__main__":
    import uvicorn
    # 设置模型和资源文件路径
    os.environ["MODEL_PATH"] = get_resource_path("models")
    """
    启动服务器的方式：
    1. 使用 uvicorn 命令（推荐）：
       uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    
    2. 直接运行此文件：
       python app/main.py
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)
