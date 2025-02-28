"""Nuitka 编译配置文件"""

# 需要包含的包列表
NUITKA_INCLUDE_PACKAGES = [
    "app",          # 项目主包
    "fastapi",      # Web框架
    "uvicorn",      # ASGI服务器
    "numpy",        # 科学计算
    "torch",        # 深度学习框架
    "torchaudio",   # 音频处理
    "librosa",      # 音频处理
    "einops",       # 张量操作
    "transformers", # 深度学习模型
    "ddsp",         # DDSP模块
    "encoder"       # 编码器模块
]

# 需要排除的包列表
NUITKA_EXCLUDE_PACKAGES = [
    "tkinter",      # 图形界面库(不需要)
    "test",         # 测试模块
    "distutils",    # 分发工具
    "pip",          # 包管理器
    "setuptools",   # 构建工具
]

# 需要包含的数据文件
NUITKA_INCLUDE_DATA_FILES = [
    ("models/*", "models"),           # 模型文件
    ("configs/*.yaml", "configs"),    # 配置文件
    ("app/assets/*", "app/assets"),   # 资源文件
]

# 编译选项
NUITKA_OPTIONS = {
    "enable_plugins": [
        "numpy",           # 启用numpy支持
        "torch",          # 启用PyTorch支持
        "multiprocessing" # 启用多进程支持
    ],
    "windows_icon": "app/assets/icon.ico",  # 应用图标
    "windows_company_name": "Jamboxx",      # 公司名称
    "windows_product_name": "Jamboxx Infinite",  # 产品名称
    "windows_file_version": "1.0.0.0",      # 文件版本
    "windows_product_version": "1.0.0.0",   # 产品版本
    "disable_console": True,                # 禁用控制台窗口
}

# 运行时配置
RUNTIME_VARS = {
    "MODEL_PATH": "models",           # 模型路径
    "CONFIG_PATH": "configs",         # 配置路径
    "LOG_LEVEL": "INFO",             # 日志级别
    "PORT": 8000                     # 服务端口
}