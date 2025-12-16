# Diversity-Aware Adaptive Collocation for Physics-Informed Neural Networks with Quantum UBO

This repository studies **combinatorial, diversity-aware collocation point selection for Physics-Informed Neural Networks (PINNs)**, with a focus on **reducing the number of training points** while maintaining or improving solution accuracy.

The core idea is to reinterpret **collocation point selection as a combinatorial optimization problem**, inspired by recent work on QUBO-based episode selection in reinforcement learning. Instead of relying solely on residual-based refinement, we explicitly balance **informativeness** and **diversity** of collocation points.

Our primary benchmark is the **1D time-dependent Burgers’ equation**, a canonical PINN test case with shock formation and localized structures.

---

## Motivation

Standard PINNs typically rely on:
- Uniform collocation sampling, or
- Residual-based adaptive refinement

However, these approaches often:
- Oversample smooth regions
- Produce highly correlated collocation points
- Scale poorly with domain resolution
- Ignore global diversity of the training set

This project explores whether **combinatorial selection** can act as a *data coreset mechanism* for PINNs.

---

## Core Idea

Given a large candidate set of collocation points, we:
1. Estimate **point importance** (e.g., PDE residual, gradient norm)
2. Measure **pairwise similarity** (e.g., spatial proximity)
3. Solve a **Quadratic Unconstrained Binary Optimization (QUBO)** problem to select a compact, diverse subset
4. Train the PINN **only on the selected points**

The QUBO objective balances:
- **Linear terms** → informative points
- **Quadratic terms** → redundancy penalties

This mirrors episode selection strategies recently proposed for Monte Carlo reinforcement learning.

---

## Governing Equation (Benchmark)

We consider the 1D viscous Burgers’ equation:


$u_t + u u_x = \nu u_{xx}, \quad x \in [-1, 1], \; t \in [0, T]$

with standard initial and boundary conditions used in PINN literature.

This problem is ideal because:
- Shocks create localized regions of high error
- Uniform sampling is inefficient
- Adaptive sampling is known to be essential

---

## Methods Compared

We compare the following collocation strategies:

1. **Uniform sampling**
2. **Random subsampling**
3. **Residual-based adaptive sampling**
4. **QUBO-based diversity-aware selection (ours)**

All methods use the same PINN architecture and optimizer.

---

## QUBO Formulation (High-Level)

Each candidate collocation point \( x_i \) is associated with a binary variable \( z_i \in \{0,1\} \).

The optimization objective is:

\[
\min_{z} \sum_i (-\alpha s_i) z_i + \sum_{i<j} \gamma w_{ij} z_i z_j
\]

where:
- \( s_i \): importance score (e.g., PDE residual)
- \( w_{ij} \): similarity between points (e.g., distance-based)
- \( \alpha, \gamma \): trade-off coefficients

The result is a **compact and diverse collocation set**.

The QUBO is solved using:
- Simulated Bifurcation (SB)
- Simulated Quantum Annealing (SQA)
- or classical heuristics (baseline)

---

## Repository Structure

├── data/
│ └── burgers_reference_solution/
├── pinn/
│ ├── model.py
│ ├── loss.py
│ └── train.py
├── sampling/
│ ├── uniform.py
│ ├── residual_adaptive.py
│ ├── qubo_selection.py
│ └── similarity_metrics.py
├── experiments/
│ ├── burgers_uniform.yaml
│ ├── burgers_adaptive.yaml
│ └── burgers_qubo.yaml
├── results/
│ ├── figures/
│ └── logs/
├── README.md
└── requirements.txt
