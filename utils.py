# File containing utilities
from typing import Callable


def reportedRun(title: str, fun: Callable[[], None]):
    print(f"{title}...")
    fun()
    print(f"{title} Done.")
