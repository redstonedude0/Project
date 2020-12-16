# File containing utilities
from typing import Callable

from hyperparameters import SETTINGS


def reportedRun(title: str, fun: Callable[[], None]):
    print(f"{title}...")
    fun()
    print(f"{title} Done.")


def debug(*args, **kwargs):
    if SETTINGS.DEBUG:
        print(*args, **kwargs)
