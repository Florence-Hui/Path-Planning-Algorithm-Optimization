import numpy as np


class HorizonScheduler:
    """
    HorizonScheduler implements an adaptive planning horizon strategy for
    Monte Carlo Tree Search (MCTS) in informative path planning.

    Instead of using a fixed rollout length (planning horizon), this class
    dynamically adjusts the horizon H_t as a function of the current belief
    uncertainty of the Gaussian Process (GP) model.

    The adaptive horizon enables the planner to:
        - Use deeper lookahead when global uncertainty is high
        - Reduce computational cost when uncertainty decreases
        - Or alternatively increase horizon as confidence improves

    Args :
    H_min : int
        Minimum allowable planning horizon.
    H_max : int
        Maximum allowable planning horizon.
    mode : str
        Strategy for scheduling the horizon.
        Options:
            - "decreasing":  H_t ∝ U_t / U_0
            - "increasing":  H_t ∝ 1 - U_t / U_0
            - "exp":         H_t ∝ exp(-gamma * (U_t / U_0))
        Default is "uncertainty".
    gamma : float
        Sensitivity parameter for exponential mode.
        Larger gamma results in faster horizon growth.
    """

    def __init__(self, H_min, H_max, mode="decreasing", gamma=1.0):
        self.H_min = H_min
        self.H_max = H_max
        self.mode = mode
        self.gamma = gamma

        # Store initial uncertainty for normalization
        self.U0 = None

    def compute_uncertainty(self, gp_model, grid):
        """
        Compute global uncertainty of the GP belief.

        Uncertainty is approximated as the sum of predictive variances
        over a discretized spatial grid.

        Args :
        gp_model : GPModel
            Current Gaussian Process belief model.
        grid : np.ndarray
            Set of spatial query points used to approximate global uncertainty.

        Returns :
        float
            Scalar global uncertainty measure U_t.
        """
        mu, var = gp_model.predict_value(grid)
        return np.sum(var)

    def get_H(self, gp_model, grid, t=None):
        """
        Compute adaptive planning horizon H_t.

        The horizon is scaled between H_min and H_max according to
        the current uncertainty ratio U_t / U_0.

        Args :
        gp_model : GPModel
            Current belief model.
        grid : np.ndarray
            Grid used to evaluate global uncertainty.
        t : int, optional
            Current time step (not required but kept for extensibility).

        Returns :
        int
            Adaptive rollout length H_t.
        """
        U_t = self.compute_uncertainty(gp_model, grid)

        print("time:", t)
        print("uncertainty:", U_t)

        # Initialize baseline uncertainty
        if self.U0 is None:
            self.U0 = U_t

        ratio = U_t / self.U0

        if self.mode == "decreasing":
            scale = ratio

        elif self.mode == "increasing":
            scale = 1.0 - ratio

        elif self.mode == "exp":
            scale = np.exp(-self.gamma * ratio)

        else:
            raise ValueError("Invalid horizon scheduling mode.")

        H = self.H_min + (self.H_max - self.H_min) * scale

        return int(np.clip(H, self.H_min, self.H_max))
