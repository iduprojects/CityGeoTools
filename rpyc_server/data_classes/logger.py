import os
import logging
import coloredlogs

os.environ["COLOREDLOGS_FIELD_STYLES"]=''
coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger(__name__)

if os.environ.get("LOG_FILE"):
    fileHandler = logging.FileHandler(os.environ["LOG_FILE"], mode='w')
    fileHandler.setLevel(logging.CRITICAL)
    logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)