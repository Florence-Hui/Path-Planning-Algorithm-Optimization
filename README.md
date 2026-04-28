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

---
## Setup Guide

This guide explains how to set up the environment and run the Adaptive Horizon Scheduling project.

---

-**1. Prerequisites**

Make sure you have the following installed:

- Python 3.8+
- pip (Python package manager)
- Git

-**2. Clone the Repository**
```bash
git clone https://github.com/Florence-Hui/Path-Planning-Algorithm-Optimization.git
cd Path-Planning-Algorithm-Optimization
```

-**3. Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

-**4. Necessary Imports**

To run the demo notebook and to use some of the libraries, the following packages are necessary:

- numpy
- scipy
- GPy (a Gaussian Process library)
- dubins (a Dubins curve generation library)
- matplotlib
- Ipython
  
-**5. Running the Code**

To run the main experiment:
```bash
python real_data_experiment.py
```
This will:
* Initialize the robot and environment
* Run MCTS-based planning
* Apply adaptive horizon scheduling
* Generate performance metrics and plots

-**6. Key Parameters to Modify**

In ```real_data_experiments.py```

* T: total number of timesteps
* number of runs (for averaging)

In ```horizon_scheduler.py```

* mode: "decreasing", "increasing", "exp"
* Sensitivity parameter for exponential mode (gamma)
* confidence thresholds
* locking conditions

In ```robot_library.py```

* spatial penalty weight (lambda)
* reward function settings




