# compute_array.py
import math
from config import Config
from pruning import nonzero_fraction

class ComputeArray:
    def __init__(self):
        self.base_macs_per_cycle = Config.MAC_UNITS

    def matmul(self, m, k, n, prune_frac, precision):
        nz = nonzero_fraction(prune_frac)
        macs = int(m * k * n * nz * Config.PRUNE_EFFICIENCY)

        density = Config.COMPUTE_DENSITY_SCALE[precision]

        utilization = min(1.0, (m * k * n) / 1e6)
        effective_macs_per_cycle = self.base_macs_per_cycle * density * utilization

        cycles = max(1, math.ceil(macs / effective_macs_per_cycle))
        return cycles, macs
