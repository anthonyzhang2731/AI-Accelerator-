# AI-Accelerator-
Analytical Investigation of Structured Pruning, Knowledge Distillation, and Quantization in a Python AI Accelerator Simulation
Overview

This project presents a Python-based AI accelerator simulator developed to analytically compare three major neural network optimization strategies:

Structured Pruning

Knowledge Distillation (KD)

Quantization

The goal of this research is to determine which method provides the best balance between energy efficiency and model accuracy when evaluated under identical hardware constraints.

Unlike many prior studies, all optimization strategies were tested under the same simulated hardware parameters to isolate the true efficiency trade-offs.

Research Question

Which AI optimization strategy — structured pruning, knowledge distillation, or quantization — provides the best balance between energy efficiency and model accuracy when evaluated on a simulated AI accelerator with fixed hardware parameters?

Motivation

Artificial intelligence systems require billions of operations per inference. Data centers now consume nearly 2% of global electricity, with AI workloads contributing significantly.

Although pruning, quantization, and knowledge distillation can each reduce model size and computation, few studies compare them under identical hardware constraints. Without hardware-controlled comparison, it is difficult to determine which strategy is truly most efficient.

This project addresses that gap using a controlled accelerator simulation.

Hardware Simulation Parameters

The simulator models a fixed AI accelerator with the following specifications:

Clock frequency: 1 GHz

64 parallel MAC (Multiply–Accumulate) units

DRAM bandwidth: 200 MB/s

Memory latency: 100 ns

Matrix multiplication workloads simulate neural network inference, including both compute-bound and memory-bound behavior.

Optimization Methods Modeled
1. Quantization

Precision reduction from FP32 to:

FP16

INT8

INT4

Lower precision reduces bytes transferred and compute cost but impacts accuracy.

2. Structured Pruning

Modeled by reducing the nonzero weight fraction of matrices.

Computation decreases proportionally to remaining nonzero weights

Metadata overhead included

Accuracy modeled as a function of pruning level

3. Knowledge Distillation (KD)

Simulated by scaling network dimensions to represent a smaller student model trained to mimic a larger model.

Reduces total MAC operations

Reduces memory transfers

Accuracy factor scaled relative to compression level

Metrics Calculated

For each configuration, the simulator computes:

Total cycles

Multiply–Accumulate (MAC) operations

Memory transfer cycles

Bytes transferred

Throughput

Estimated energy consumption

Predicted accuracy factor

An efficiency score was defined as:

Efficiency = Accuracy / Energy

This ratio identifies the best trade-off between performance and energy usage.

Key Result

The configuration that achieved the highest efficiency score was:

50% Knowledge Distillation + INT4 Quantization

Efficiency Score: 13.4875

Although this configuration is memory-intensive, it provides the strongest accuracy-to-energy ratio under the modeled hardware constraints.

For this specific AI accelerator architecture, the combination of moderate distillation with aggressive quantization produced the most favorable trade-off.

Project Structure

The repository includes:

Hardware configuration model

Compute system modeling MAC parallelism

Memory system modeling bandwidth and latency

Optimization parameter modeling

Workload simulation via matrix multiplication

Graph generation scripts (graphs.py)

Performance analysis outputs

Visualizations Generated

All figures were generated programmatically using Python:

Performance comparison table

Bar chart of efficiency scores

Energy vs. Accuracy plot

Roofline analysis

Energy breakdown

Hardware utilization plot

Use of AI Tools

All aspects of the project were designed, implemented, and analyzed independently.

Large language models were used only as assistants to:

Explain coding concepts

Clarify theoretical foundations

Improve phrasing of written explanations

All simulator design, modeling assumptions, implementation, and interpretation of results were performed independently.

Help Received

No outside assistance was used.

Bibliography

von Rad, Jonathan, Yong Cao, and Andreas Geiger. “UNICOMP: A Unified Evaluation of Large Language Model Compression via Pruning, Quantization, and Distillation.” arXiv, 2026.

Sander, Jacob, et al. “On Accelerating Edge AI: Optimizing Resource-Constrained Environments.” arXiv, 2025.

González, Alexandra, et al. “Impact of ML Optimization Tactics on Greener Pre-trained ML Models.” Computing, 2025.

Qu, Xiaoyi, et al. “Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression.” arXiv, 2025.

Harma, Simla B., et al. “Effective Interplay between Sparsity and Quantization.” arXiv, 2024.

Muralidharan, Saurav, et al. “Compact Language Models via Pruning and Knowledge Distillation.” arXiv, 2024.

Kuzmin, Andrey, et al. “Pruning vs Quantization: Which Is Better?” arXiv, 2024.

Wang, Wenxiao, et al. “Model Compression and Efficient Inference for Large Language Models: A Survey.” arXiv, 2024.
