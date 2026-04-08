# Simulation-Based Safe Reinforcement Learning for Energy Optimization of a Distillation Column

This repository contains a **simulation-based proof-of-principle** implementation of *safe reinforcement learning (Safe RL)* for control of a **continuous distillation column**. The control objective is to **reduce reboiler energy demand (HeatDuty)** while maintaining **product purity and safety constraints**, most importantly the engineering requirement:

- **Distillate purity constraint:** **XD ≥ 0.95**

The work is intended as a **research / academic / portfolio project** demonstrating the integration of **process engineering**, **control**, and **reinforcement learning** in a modular Python codebase. It is **not** a real plant controller and **not** a real-time industrial deployment.

---

## Requirements

Main Python dependencies used in this project:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `gymnasium`
- `stable-baselines3`
- `scikit-learn`

---

## Architecture / workflow (high level)

```text
Aspen Plus Data
→ Data Cleaning
→ Surrogate Model
→ FOPDT Dynamics
→ Safe RL Environment
→ SAC Training
→ Evaluation vs PID/Random
→ Robustness Analysis
```

---

## What this project does

At a high level, the project implements and evaluates a closed-loop control workflow:

1. **Loads and cleans** steady-state process data generated from **Aspen Plus** simulations.
2. Builds a **steady-state surrogate model** for the column behavior from the dataset:
   - Interpolation-based surrogate (data-driven)
   - MLP-based surrogate (neural network regression)
3. Approximates process dynamics using a **First-Order-Plus-Dead-Time (FOPDT)** model.
   - This is an **engineering approximation** and **not** a full first-principles dynamic distillation model.
4. Wraps the surrogate + FOPDT approximation into a **Gym-style RL environment**.
5. Trains a **Soft Actor-Critic (SAC)** policy to manipulate **HeatDuty** (single manipulated variable).
6. Adds a **safety layer** to enforce/handle constraints (purity and temperature limits).
7. Compares SAC to baseline policies (PID and random) and performs **robustness analysis** including stress tests, disturbances, and noise.

---

## Project scope and limitations

**Scope (what is covered):**
- Closed-loop, simulation-based Safe RL for a distillation column surrogate
- Single-input control (manipulated variable: **HeatDuty**)
- Primary constraint: **XD ≥ 0.95**, plus additional safety limits (e.g., temperature constraints)
- End-to-end pipeline: data → surrogate → dynamic approximation → environment → training → evaluation → robustness

**Limitations (what this is not):**
- **Not** a plant-ready controller and **not** connected to any DCS/PLC hardware
- **Not** real-time validated; all results are within a simulated environment built from Aspen-based steady-state data and simplified dynamics
- The **FOPDT** component is a practical approximation; it does not capture all nonlinearities and multivariable dynamics of a real column
- The surrogate model is based on **a specific Aspen dataset**; it should not be assumed to generalize to all distillation systems without additional modeling and validation

---

## Repository structure

The repository is organized as a set of modular “level-based” Python files:

- `main_Version.py` — Entry point / orchestration script (typical starting point)
- `level0_config.py` — Central configuration (paths, parameters, constraints, toggles)
- `level1_data_layer.py` — Data loading, preprocessing, cleaning utilities
- `level2_surrogate.py` — Surrogate modeling (interpolation and MLP options)
- `level3_feasibility.py` — Feasibility analysis and constraint-related checks
- `level4_dynamics.py` — Dynamic approximation via FOPDT (engineering approximation)
- `level5_safety.py` — Safety layer implementation (purity/temperature constraints)
- `level6_env_core.py` — Environment core logic (state transitions, rewards, constraints)
- `level7_gym_wrapper.py` — Gym-style wrapper interface for RL training
- `level8_sanity_checks.py` — Consistency checks / validation utilities for simulation setup
- `level9_training.py` — SAC training pipeline
- `level10_evaluation.py` — Policy evaluation (SAC + baselines)
- `level11_reporting.py` — Reporting utilities (tables/metrics summaries as implemented)
- `level12_plots.py` — Plotting utilities for training/evaluation/robustness outputs
- `level13_robustness.py` — Robustness tests (disturbances, noise, stress tests)
- `level14_advanced.py` — Advanced / extended analyses (as implemented)
- `level15_pid_baseline.py` — PID baseline controller for comparison

---

## Method overview

**Control objective**
- Minimize **reboiler energy demand** (HeatDuty) while maintaining **purity constraint**:
  - **XD ≥ 0.95**

**Manipulated variable (MV)**
- **HeatDuty only** (single-input policy/controller)

**Process modeling approach**
- **Steady-state basis:** Aspen Plus simulation dataset  
- **Surrogate models (steady-state):**
  1. Interpolation-based surrogate (data-driven mapping)
  2. MLP-based surrogate (regression model)
- **Dynamic approximation:** FOPDT-based dynamics layered on top of the steady-state surrogate  
  - Used to emulate closed-loop behavior in time
  - Intended as an engineering approximation suitable for proof-of-principle studies

**Reinforcement learning**
- **Algorithm:** Soft Actor-Critic (SAC)
- **Environment style:** Gym-like interface
- **Safety handling:** safety layer enforcing purity and temperature limits (implementation-specific)

---

## Implemented components (technical)

This repository includes a complete workflow typically needed for a safe RL study in process control:

- **Data loading and cleaning** (Aspen-derived dataset handling)
- **Surrogate modeling**
  - Interpolation-based surrogate option
  - MLP-based surrogate option
- **Feasibility analysis**
  - Constraint feasibility checks (e.g., purity-related feasibility regions)
- **Dynamic approximation**
  - FOPDT model used to approximate time-domain behavior
- **Safety layer**
  - Purity constraint enforcement (XD ≥ 0.95)
  - Temperature limit handling (as defined in the project configuration)
- **Gym-style environment**
  - State, action, transition, reward, termination logic
- **SAC training**
  - Training loop, logging, and saved artifacts (as implemented)
- **Evaluation and reporting**
  - Comparative evaluation utilities and reporting scripts
- **Plotting**
  - Training curves, trajectories, constraint-related plots (as implemented)
- **Robustness analysis**
  - Stress tests, disturbances, measurement/process noise experiments
- **Baseline controllers**
  - PID baseline
  - Random policy baseline

---

## Baselines and evaluation

The project evaluates the learned SAC policy against:

- **PID baseline** (classical feedback control comparison)
- **Random baseline** (sanity check for RL learning signal)

Evaluation focuses on the trade-off between:
- **Energy usage (HeatDuty)** reduction
- **Constraint adherence** (XD ≥ 0.95 and defined temperature limits)
- Performance under **robustness tests** (disturbances, noise, stress conditions)

*Note:* This README intentionally does not report numerical performance claims; consult the scripts and generated outputs in your local runs for experiment-specific results.

---

## Why this project is relevant

Distillation remains one of the most energy-intensive unit operations in the process industries. This project is relevant because it demonstrates, in a controlled research setting:

- How to translate **process simulation data** into a usable control-oriented surrogate
- How to integrate **constraint handling** into an RL workflow (Safe RL motivation)
- How to implement and evaluate an RL controller (SAC) against **classical baselines** (PID)
- How to perform **robustness testing** (disturbances/noise) rather than relying only on nominal conditions

Overall, it serves as an example of **applied reinforcement learning for process control** with explicit attention to engineering constraints.

---

## Future work

Realistic extensions that would strengthen the study include:

- Replace/augment FOPDT dynamics with a more detailed dynamic model (e.g., higher-order, nonlinear, or first-principles dynamic simulation when available)
- Extend from single-input (**HeatDuty**) to a multivariable setting (e.g., reflux, boilup, feed conditions) when appropriate
- More explicit constraint handling methods (e.g., CMDPs, Lagrangian methods, barrier methods) and comparisons against a safety-layer approach
- Systematic uncertainty quantification for the surrogate (e.g., ensembles, Bayesian methods) and its impact on safe control
- Cross-validation across different Aspen cases / operating windows to assess generalization
- Reproducible experiment management (seed control, config snapshots, result directories) if not already present

---

## How to run / adapt (minimal)

A typical workflow is:

1. Review and adjust settings in `level0_config.py` (paths, constraints, model choices).
2. Use `main_Version.py` to run the pipeline end-to-end (data → surrogate → environment → training → evaluation), or run individual levels directly (e.g., `level9_training.py` for training).
3. Use:
   - `level10_evaluation.py` for policy evaluation
   - `level12_plots.py` for visualization
   - `level13_robustness.py` for robustness/stress testing
   - `level15_pid_baseline.py` for baseline comparisons

Because this is a research-style repository, exact run steps may depend on your local environment and data paths. The code is structured to be adapted: swap surrogate type (interpolation vs MLP), modify constraints, or adjust the FOPDT approximation and reward definitions while keeping the overall pipeline intact.

---

## Notes

- This repository is **private** at the time of writing (2026-04-07) and intended for research/portfolio use.
- Aspen Plus is used as the steady-state data source; the repository uses the resulting dataset, not a live Aspen integration.


## Repository purpose

This repository is intended as a research / portfolio project
demonstrating the integration of:

- process engineering
- control systems
- reinforcement learning

in a simulation-based study of distillation column energy optimization.


## License

This repository is provided for research and portfolio purposes.

All rights reserved.  
The code may not be copied, redistributed, or used in other projects without explicit permission from the author.

