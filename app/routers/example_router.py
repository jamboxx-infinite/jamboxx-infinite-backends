from fastapi import APIRouter
from app.services import example_service

router = APIRouter(prefix="/api/v1")

@router.get("/example")
async def get_example():
    """示例API端点"""
    return example_service.get_example_data()
