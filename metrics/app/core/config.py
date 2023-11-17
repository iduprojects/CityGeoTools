from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application global config class, attributes can be modified via environment variables."""

    BASE_DIR: str = str(Path(__file__).resolve().parent.parent.parent)
    """Metrics module base directory (CityGeoTools/metrics)."""

    DATA_DIR: str = str(Path(__file__).resolve().parent.parent.parent.parent) # seems to be unused
    """Whole project base directory."""

    CITIES_CACHE_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "cities_cache")
    """Directory with pickle files of cities data to fasten metrics startup process."""

    UPDATE_CITIES_CACHE: bool = False
    """Indicates whether cached cities need to be re-downloaded on startup."""

    FASTAPI_DEBUG: bool = True
    """FastAPI debug flag."""


settings = Settings()
"""Settings singleton to use across application."""