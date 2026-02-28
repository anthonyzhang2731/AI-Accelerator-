from config import Config
from compute_array import ComputeArray
from memory_system import MemorySystem
from workload import load_workload
from kd import apply_kd  # <- make sure this is imported


# --- Key methods to keep, including ONE pruning method ---
METHODS = [
   {"name": "Default FP16", "prune_frac": 0.0, "kd_scale": None, "precision": "FP16"},
   {"name": "Quant INT8", "prune_frac": 0.0, "kd_scale": None, "precision": "INT8"},
   {"name": "Quant INT4", "prune_frac": 0.0, "kd_scale": None, "precision": "INT4"},
   {"name": "KD 50% + INT4", "prune_frac": 0.0, "kd_scale": 0.5, "precision": "INT4"},
   {"name": "Prune 50% FP16", "prune_frac": 0.5, "kd_scale": None, "precision": "FP16"},
]


def run(prune_frac=0.0, kd_scale=None, precision=None):
   if kd_scale is not None:
       Config.KD_STUDENT_SCALE = kd_scale
   if precision is not None:
       Config.DEFAULT_PRECISION = precision


   workload = load_workload()
   array = ComputeArray()
   memory = MemorySystem()


   total_cycles = 0
   total_macs = 0
   total_bytes = 0


   for op in workload:
       # ---- APPLY KD SCALE BEFORE MATMUL ----
       if kd_scale is not None:
           m_eff, k_eff, n_eff = apply_kd(op["m"], op["k"], op["n"])
       else:
           m_eff, k_eff, n_eff = op["m"], op["k"], op["n"]


       c_compute, macs = array.matmul(
           m_eff, k_eff, n_eff,
           prune_frac,
           precision
       )
       c_memory = memory.transfer_cycles(op["bytes"])
       cycles = max(c_compute, c_memory)


       total_cycles += cycles
       total_macs += macs
       total_bytes += op["bytes"]


   # ---------------------------
   # Energy
   # ---------------------------
   mac_energy = (
       total_macs *
       Config.ENERGY_PER_MAC_FP16 *
       Config.MAC_ENERGY_SCALE[Config.DEFAULT_PRECISION]
   )
   dram_energy = total_bytes * Config.ENERGY_PER_BYTE_DRAM
   total_energy = mac_energy + dram_energy


   # ---------------------------
   # Accuracy Model
   # ---------------------------
   BASELINE_ACCURACY = 1.0


   if prune_frac > 0:
       # Nonlinear pruning degradation
       accuracy = max(0.1, BASELINE_ACCURACY - prune_frac**2.2)
   else:
       accuracy = BASELINE_ACCURACY


   if kd_scale is not None:
       accuracy *= kd_scale


   # ---------------------------
   # Output (keep same structure)
   # ---------------------------
   output = total_macs


   # ---------------------------
   # Score (simplified & scaled for readability)
   # ---------------------------
   score = (accuracy / total_energy) * 1e7


   # ---------------------------
   # Throughput
   # ---------------------------
   throughput = total_macs / (total_cycles / Config.CLOCK_HZ)


   return {
       "Cycles": total_cycles,
       "MACs": total_macs,
       "Bytes": total_bytes,
       "Throughput": throughput,
       "Score": score
   }


def main():
   results = []


   for method in METHODS:
       results.append((
           method["name"],
           run(
               prune_frac=method["prune_frac"],
               kd_scale=method["kd_scale"],
               precision=method["precision"]
           )
       ))


   # Sort by score descending
   results.sort(key=lambda x: x[1]["Score"], reverse=True)


   # Print clean comparison table
   print("\nComparison Table:")
   header = ["Method", "Cycles", "MACs", "Bytes", "Throughput", "Score"]
   print("-"*90)
   print(f"{header[0]:<22}{header[1]:>10}{header[2]:>12}{header[3]:>12}"
         f"{header[4]:>14}{header[5]:>12}")
   print("-"*90)


   for name, r in results:
       print(f"{name:<22}{r['Cycles']:>10,}{r['MACs']:>12,}{r['Bytes']:>12,}"
             f"{r['Throughput']:>14.2e}{r['Score']:>12.4f}")


if __name__ == "__main__":
   main()



