from config import Config
from compute_array import ComputeArray
from memory_system import MemorySystem
from workload import load_workload
from kd import apply_kd


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# -------------------------------
# Methods & color map
# -------------------------------
METHODS = [
   {"name": "Default FP16", "prune_frac": 0.0, "kd_scale": None, "precision": "FP16"},
   {"name": "Quant INT8", "prune_frac": 0.0, "kd_scale": None, "precision": "INT8"},
   {"name": "Quant INT4", "prune_frac": 0.0, "kd_scale": None, "precision": "INT4"},
   {"name": "KD 50% + INT4", "prune_frac": 0.0, "kd_scale": 0.5, "precision": "INT4"},
   {"name": "Prune 50% FP16", "prune_frac": 0.5, "kd_scale": None, "precision": "FP16"},
]


COLOR_MAP = {
   "KD 50% + INT4": "#ff7f0e",
   "Quant INT4": "#1f77b4",
   "Prune 50% FP16": "#9467bd",
   "Quant INT8": "#2ca02c",
   "Default FP16": "#d62728",
}


# -------------------------------
# Run simulation
# -------------------------------
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
       if kd_scale is not None:
           m_eff, k_eff, n_eff = apply_kd(op["m"], op["k"], op["n"])
       else:
           m_eff, k_eff, n_eff = op["m"], op["k"], op["n"]


       c_compute, macs = array.matmul(m_eff, k_eff, n_eff, prune_frac, precision)
       c_memory = memory.transfer_cycles(op["bytes"])
       cycles = max(c_compute, c_memory)


       total_cycles += cycles
       total_macs += macs
       total_bytes += op["bytes"]


   mac_energy = total_macs * Config.ENERGY_PER_MAC_FP16 * Config.MAC_ENERGY_SCALE[Config.DEFAULT_PRECISION]
   dram_energy = total_bytes * Config.ENERGY_PER_BYTE_DRAM
   total_energy = mac_energy + dram_energy


   BASELINE_ACCURACY = 1.0
   if prune_frac > 0:
       accuracy = max(0.1, BASELINE_ACCURACY - prune_frac**2.2)
   else:
       accuracy = BASELINE_ACCURACY


   if kd_scale is not None:
       accuracy *= kd_scale


   score = (accuracy / total_energy) * 1e7
   throughput = total_macs / (total_cycles / Config.CLOCK_HZ)


   return {
       "Cycles": total_cycles,
       "MACs": total_macs,
       "Bytes": total_bytes,
       "Throughput": throughput,
       "Score": score,
       "Energy": total_energy,
       "Accuracy": accuracy,
       "MAC_Energy": mac_energy,
       "DRAM_Energy": dram_energy
   }


# -------------------------------
# Main plotting
# -------------------------------
def main():
   results = []
   for method in METHODS:
       results.append((
           method["name"],
           run(prune_frac=method["prune_frac"],
               kd_scale=method["kd_scale"],
               precision=method["precision"])
       ))


   results.sort(key=lambda x: x[1]["Score"], reverse=True)


   # Print comparison table
   print("\nComparison Table:")
   print("-"*90)
   print(f"{'Method':<22}{'Cycles':>10}{'MACs':>12}{'Bytes':>12}{'Throughput':>14}{'Score':>12}")
   print("-"*90)
   for name, r in results:
       print(f"{name:<22}{r['Cycles']:>10,}{r['MACs']:>12,}{r['Bytes']:>12,}"
             f"{r['Throughput']:>14.2e}{r['Score']:>12.4f}")


   # Prepare arrays for plotting
   methods = [x[0] for x in results]
   colors = [COLOR_MAP[m] for m in methods]


   cycles = np.array([x[1]["Cycles"] for x in results])
   macs = np.array([x[1]["MACs"] for x in results])
   bytes_moved = np.array([x[1]["Bytes"] for x in results])
   throughput = np.array([x[1]["Throughput"] for x in results])
   scores = np.array([x[1]["Score"] for x in results])
   energy = np.array([x[1]["Energy"] for x in results])
   accuracy = np.array([x[1]["Accuracy"] for x in results])
   mac_energy = np.array([x[1]["MAC_Energy"] for x in results])
   dram_energy = np.array([x[1]["DRAM_Energy"] for x in results])


   # -------------------------------
   # 1️⃣ Composite Score
   # -------------------------------
   plt.figure(figsize=(8,5))
   bars = plt.bar(methods, scores, color=colors)
   plt.ylabel("Score (Accuracy / Energy × 1e7)")
   plt.title("AI Accelerator Optimization: Score by Method")
   plt.ylim(0, max(scores)*1.2)
   plt.xticks(rotation=45)
   for bar, score in zip(bars, scores):
       plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{score:.4f}", ha='center', va='bottom')
   plt.tight_layout()
   plt.show()


   # -------------------------------
   # 2️⃣ Energy vs Accuracy
   # -------------------------------
   plt.figure(figsize=(8,6))
   plt.scatter(energy, accuracy, s=150, c=colors)
   for i, m in enumerate(methods):
       plt.annotate(m, (energy[i], accuracy[i]))
   plt.xlabel("Total Energy")
   plt.ylabel("Accuracy")
   plt.title("Energy vs Accuracy Tradeoff")
   plt.grid(True)
   plt.tight_layout()
   plt.show()


   # -------------------------------
   # 3️⃣ Roofline Style
   # -------------------------------
   arith_intensity = macs / bytes_moved
   plt.figure(figsize=(8,6))
   plt.scatter(arith_intensity, throughput, s=150, c=colors)
   for i, m in enumerate(methods):
       plt.annotate(m, (arith_intensity[i], throughput[i]))
   plt.xscale("log")
   plt.yscale("log")
   plt.xlabel("Arithmetic Intensity (MACs / Byte)")
   plt.ylabel("Throughput (MACs/sec)")
   plt.title("Roofline-Style Performance")
   plt.grid(True, which="both", ls="--")
   plt.tight_layout()
   plt.show()


   # -------------------------------
   # 4️⃣ Energy Breakdown (method color + gray DRAM)
   # -------------------------------
   x = np.arange(len(methods))
   plt.figure(figsize=(9,6))
   plt.bar(x, mac_energy, color=colors, label="Compute (MAC) Energy")
   plt.bar(x, dram_energy, bottom=mac_energy, color="gray", label="DRAM Energy")
   plt.xticks(x, methods, rotation=45)
   plt.ylabel("Energy")
   plt.title("Energy Breakdown by Method")


   # Legend: top-left now
   legend_elements = [Patch(facecolor=colors[i], label=methods[i]) for i in range(len(methods))]
   legend_elements.append(Patch(facecolor="gray", label="DRAM Energy"))
   plt.legend(handles=legend_elements, loc="upper left")


   plt.tight_layout()
   plt.show()


   # -------------------------------
   # 5️⃣ Hardware Utilization
   # -------------------------------
   macs_per_cycle = macs / cycles
   bytes_per_cycle = bytes_moved / cycles
   plt.figure(figsize=(8,6))
   plt.scatter(macs_per_cycle, bytes_per_cycle, s=150, c=colors)
   for i, m in enumerate(methods):
       plt.annotate(m, (macs_per_cycle[i], bytes_per_cycle[i]))
   plt.xlabel("MACs per Cycle")
   plt.ylabel("Bytes per Cycle")
   plt.title("Hardware Utilization")
   plt.grid(True)
   plt.tight_layout()
   plt.show()




if __name__ == "__main__":
   main()



