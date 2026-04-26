# Adaptive Horizon Scheduling for MCTS-based Informative Path Planning

This repository implements an adaptive planning framework for **Monte Carlo Tree Search (MCTS)** in informative path planning under uncertainty. The project focuses on improving the stability and performance of exploration–exploitation trade-offs through **adaptive horizon scheduling** and **reward–uncertainty integration**.

---

## Overview

Planning under uncertainty is a core challenge in robotics, especially in domains such as environmental monitoring and autonomous exploration. While MCTS is a powerful planning method, its performance is highly sensitive to the **planning horizon (H)**.

This project explores and improves adaptive horizon strategies by:

- Investigating baseline adaptive modes:
  - Increasing horizon
  - Decreasing horizon
  - Exponential decay
- Developing a **continuous adaptive framework** that integrates:
  - Model uncertainty
  - Reward-based confidence
  - Spatial regularization during exploitation

---

## Key Features

- **Adaptive Horizon Scheduling**
  - Dynamically adjusts planning horizon based on uncertainty and reward-based confidence
  - Supports multiple scaling modes: decreasing, increasing, exponential

- **Phase-Switching Mechanism**
  - Switched between exploration and exploitation stages controlled via confidence thresholds
  - Double-threshold design reduces unstable switching

- **Confidence-Based Smoothing and Locking**
  - Temporal smoothing using a confidence window

- **Reward-Based Adaptive Control**
  - Horizon scaling informed by ratio between current and predicted maximum

- **Spatial Regularization (Trajectory Penalty)**
  - Penalizes trajectories that move away from the current best location
  - Encourages local refinement during exploitation stage

- **Adaptive Penalty Strategy**
  - The penalty weight increases over time and decreases with uncertainty
  - Improves stability while avoiding premature convergence
     
- **Multi-run evaluation with average regret visualization**
- **Support for MCTS and POMCPOW comparison, although POMCPOW is not that ideal :(**

