import sys
import logging
from logging import StreamHandler

from pydantic import BaseSettings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = StreamHandler(stream=sys.stdout)
logger.addHandler(handler)


class TestingSettings(BaseSettings):
    APP_ADDRESS_FOR_TESTING: str = "127.0.0.1:5000/api/v2"

    SPB_MUNICIPALITIES: str = 'http://10.32.1.101:1244/api/city/1/municipalities'


testing_settings = TestingSettings()
logger.info(f"Настройки для тестирования: {testing_settings}")
