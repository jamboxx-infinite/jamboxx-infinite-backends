from app.utils import file_utils

def get_example_data():
    """获取示例数据"""
    return {
        "message": "This is an example service",
        "status": "success",
        "file_info": file_utils.get_file_info()
    }
