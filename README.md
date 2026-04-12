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

- Adaptive horizon scheduling (time + uncertainty driven)
- Continuous exploration–exploitation transition (no hard switching)
- Reward-based spatial penalty for stable exploitation
- Confidence-based phase switching with smoothing
- Confidence lock mechanism for late-stage stability
- Multi-run evaluation with average regret visualization
- Support for MCTS and POMCPOW comparison

