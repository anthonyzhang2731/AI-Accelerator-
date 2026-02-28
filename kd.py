# kd.py
from config import Config

def apply_kd(m, k, n):
    if not Config.KD_ENABLED:
        return m, k, n

    s = Config.KD_STUDENT_SCALE
    return max(1, int(m * s)), max(1, int(k * s)), max(1, int(n * s))
