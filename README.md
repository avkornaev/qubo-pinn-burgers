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

Each candidate collocation point $x_i$ is associated with a binary variable $z_i \in \{0,1\}$.

The optimization objective is:

$\min_{z} \sum_i (-\alpha s_i) z_i + \sum_{i<j} \gamma w_{ij} z_i z_j$

where:
$s_i$ is the importance score (e.g., PDE residual), $w_{ij}$ is the similarity between points (e.g., distance-based),  $\alpha, \gamma$ are the trade-off coefficients

The result is a **compact and diverse collocation set**.

The QUBO is solved using:
- Simulated Bifurcation (SB)
- Simulated Quantum Annealing (SQA)
- or classical heuristics (baseline)

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E.  
   **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.**  
   *Journal of Computational Physics*, 378, 686–707, 2019.

2. Wu, C., Zhu, M., Tan, X., Kartha, Y., & Karniadakis, G. E.  
   **A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks.**  
   *Computer Methods in Applied Mechanics and Engineering*, 406, 115818, 2023.

3. Gao, H., Wang, L., & Sun, L.  
   **Failure-informed adaptive sampling for physics-informed neural networks.**  
   *SIAM Journal on Scientific Computing*, 45(1), A1–A26, 2023.

4. Mao, Z., Jagtap, A. D., & Karniadakis, G. E.  
   **Physics-informed neural networks for high-speed flows with adaptive sampling.**  
   *Applied Mathematics and Mechanics*, 44, 1–22, 2023.

5. Tang, K., Liu, Y., & Perdikaris, P.  
   **Adversarial adaptive sampling for physics-informed neural networks.**  
   *International Conference on Learning Representations (ICLR)*, 2024.

6. Salloum, H., Jnadi, A., Kholodov, Y., & Gasnikov, A.  
   **Quantum-inspired episode selection for Monte Carlo reinforcement learning via QUBO optimization.**  
   *Proceedings of Machine Learning Research (PMLR)*, 2025.

7. Goto, H., Tatsumura, K., & Dixon, A. R.  
   **Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems.**  
   *Science Advances*, 5(4), eaav2372, 2019.

8. Crosson, E., & Harrow, A. W.  
   **Simulated quantum annealing can be exponentially faster than classical simulated annealing.**  
   *FOCS*, IEEE, 2016.


## Repository Structure
```text
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
```


