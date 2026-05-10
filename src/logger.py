"""
logger.py - Sets up how we print messages in our project.

We use a library called 'loguru' instead of regular print().
It prints nice colored messages to the screen and also saves
all messages to a file so we can look at them later if something goes wrong.
"""

from loguru import logger
from pathlib import Path
import sys



def setup_logger():
    """Sets up the logger and returns it.

    It creates a folder called 'logs' if it doesn't exist.
    Then it tells loguru to do two things:
    1. Print important messages (INFO) to the screen with colors.
    2. Save all messages (DEBUG and up) to a file. It creates a new file
       when the current one gets too big.

    Returns:
        The ready-to-use logger.
    """

    log_dir = Path("logs")

    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / "app_{time:YYYY-MM-DD}.log"

    logger.remove()


    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:MM:SS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    logger.add(
        str(log_file_path),
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        encoding="utf-8"
    )


    return logger

log = setup_logger()

if __name__ == "__main__":
    log.info("The application has been launched.")
    log.debug("This is a debug message (it only appears in the file).")
    log.warning("Attention! An alert has been generated.")
    log.error("Error: Database connection failed.")
