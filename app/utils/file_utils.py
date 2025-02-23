import os
from datetime import datetime

def get_file_info():
    """获取文件信息的工具函数"""
    return {
        "current_time": datetime.now().isoformat(),
        "app_directory": os.path.dirname(os.path.dirname(__file__))
    }
