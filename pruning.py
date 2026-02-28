# pruning.py
import math
from config import Config


def nonzero_fraction(prune_frac):
   return max(0.0, 1.0 - prune_frac)# prevents invalid result if invalid prune fraction is used

def pruning_metadata_bytes(k, n):
   blocks = math.ceil(k / Config.PRUNE_BLOCK_K)# pruning block is the removal of the entire structure group of parameters. math.ceil rounds up since partial block still needs metadata
   bits = blocks * n * Config.PRUNE_METADATA_BITS
   return (bits + 7) // 8

