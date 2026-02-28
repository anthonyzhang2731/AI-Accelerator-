# memory_system.py
import math
from config import Config

class MemorySystem:
    def __init__(self):
        self.bytes_per_cycle = Config.DRAM_BANDWIDTH_BYTES_PER_SEC / Config.CLOCK_HZ

    def transfer_cycles(self, byte_count):
        # realistic: at least 1 cycle
        return max(1, math.ceil(byte_count / self.bytes_per_cycle))
