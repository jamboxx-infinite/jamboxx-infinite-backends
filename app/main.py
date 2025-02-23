from fastapi import FastAPI
from app.routers import example_router

app = FastAPI(title="Jamboxx Backend API")

# 注册路由
app.include_router(example_router.router)

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
