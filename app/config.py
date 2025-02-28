import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = os.getenv("MODEL_PATH", BASE_DIR / "models")
    PORT = int(os.getenv("PORT", "8000"))