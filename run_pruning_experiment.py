from workload import load_workload
from compute_array import ComputeArray
from memory_system import MemorySystem
from config import Config
import pandas as pd


# ----- Experiment Setup -----
prune_levels = [0.5]   # ONLY investigating 50% pruning
precision = Config.DEFAULT_PRECISION
M = K = N = 1024       # GEMM dimensions


# Results table
results = []


# Memory and compute objects
memory = MemorySystem()
array = ComputeArray()


# ----- Helper function for energy -----
def compute_energy(macs, m, k, n, precision):
    # Compute energy
    e_mac = Config.ENERGY_PER_MAC_FP16 * Config.MAC_ENERGY_SCALE[precision]
    compute_energy = macs * e_mac

    # Memory energy estimate (dense, conservative assumption)
    memory_bytes = (m * k + k * n + m * n)
    memory_energy = memory_bytes * Config.ENERGY_PER_BYTE_DRAM

    return compute_energy + memory_energy


# ----- Run experiment -----
for p in prune_levels:

    # Load workload
    W, X = load_workload(M=M, K=K, N=N, prune_ratio=p)

    # Run compute
    cycles, macs = array.matmul(
        M, K, N,
        prune_frac=p,
        precision=precision
    )

    # Energy calculation
    total_energy = compute_energy(macs, M, K, N, precision)

    # -------- Improved Accuracy Model --------
    # Nonlinear degradation (more realistic)
    accuracy = max(0.1, 1.0 - p**2.2)

    # Output size (unchanged)
    output = M * N

    # Ratios (efficiency removed)
    energy_accuracy = total_energy / accuracy
    energy_output = total_energy / output

    # Combined score (efficiency removed)
    score = (
        1 / energy_accuracy +
        1 / energy_output
    )

    results.append({
        "Prune %": int(p * 100),
        "MACs": macs,
        "Energy": total_energy,
        "Accuracy": accuracy,
        "Output": output,
        "E/Acc": energy_accuracy,
        "E/Out": energy_output,
        "Score": score
    })


# ----- Results table -----
df = pd.DataFrame(results)
print(df.to_string(index=False, float_format="{:,.2f}".format))
