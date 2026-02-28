# config.py


class Config:
   # Clock
   CLOCK_HZ = 1_000_000_000  # 1 GHz, base frequency of the chip AKA how many MAC cycles per second


   # Compute
   MAC_UNITS = 64  # scaled down from 256 to allow memory-bound ops to appear


   # Memory
   DRAM_BANDWIDTH_BYTES_PER_SEC = 200e6  # scaled down from 200 GB/s to 200 MB/s
   DRAM_LATENCY_NS = 100 # delay before data is transferred in nanoseconds


   # Precision (bitwidth per element)
   BITS_PER_ELEMENT = {
       "FP16": 16,
       "INT9": 9,
       "INT8": 8,
       "INT4": 4,
   }


   DEFAULT_PRECISION = "FP16"


   # Quantization scaling (relative to FP16)
   MAC_ENERGY_SCALE = {
       "FP16": 1.0,
       "INT8": 0.25,
       "INT4": 0.0625,
   }


   # How many MACs can be packed per cycle
   COMPUTE_DENSITY_SCALE = {
       "FP16": 1.0,
       "INT8": 2.0,
       "INT4": 4.0,
   }


   # Pruning
   PRUNE_BLOCK_K = 8
   PRUNE_EFFICIENCY = 0.9
   PRUNE_METADATA_BITS = 1


   # Knowledge Distillation
   KD_ENABLED = True
   KD_STUDENT_SCALE = 1.0


   # Energy (normalized, relative)
   ENERGY_PER_MAC_FP16 = 1.0
   ENERGY_PER_BYTE_DRAM = 10.0

