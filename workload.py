# workload.py
from config import Config
from pruning import nonzero_fraction, pruning_metadata_bytes
from kd import apply_kd


def bytes_for_tensor(elems, bits):
   return (int(elems) * bits + 7) // 8


def matmul_bytes(m, k, n, precision, prune_frac):
   bits = Config.BITS_PER_ELEMENT[precision]
   nz = nonzero_fraction(prune_frac)
   A = bytes_for_tensor(m * k * nz, bits)
   B = bytes_for_tensor(k * n * nz, bits)
   C = bytes_for_tensor(m * n, bits)
   meta = pruning_metadata_bytes(k, n)
   return A + B + C + meta


def load_workload():
   # Classic roofline needs ops with **diverse operational intensity**
   ops = [
       # Memory-bound ops (low MACs, relatively high bytes)
       {"type":"matmul","m":16,"k":128,"n":16,"prune":0.0},
       {"type":"matmul","m":32,"k":128,"n":32,"prune":0.0},
       # Transition ops
       {"type":"matmul","m":64,"k":128,"n":64,"prune":0.0},
       # Compute-bound ops (high MACs)
       {"type":"matmul","m":128,"k":128,"n":128,"prune":0.0},
       {"type":"matmul","m":256,"k":256,"n":256,"prune":0.0},
   ]


   for op in ops:
       m, k, n = apply_kd(op["m"], op["k"], op["n"])
       op["m"], op["k"], op["n"] = m, k, n
       op["precision"] = Config.DEFAULT_PRECISION
       op["bytes"] = matmul_bytes(m, k, n, op["precision"], op["prune"])


   return ops

