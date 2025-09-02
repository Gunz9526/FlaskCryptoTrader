import logging
from pythonjsonlogger import jsonlogger

def setup_logging():
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s %(module)s %(lineno)d'
    )
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)