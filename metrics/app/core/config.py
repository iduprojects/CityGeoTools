from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    BASE_DIR: str = str(Path(__file__).resolve().parent.parent.parent)
    DATA_DIR: str = str(Path(__file__).resolve().parent.parent.parent.parent)

    FASTAPI_DEBUG: bool = True


settings = Settings()
