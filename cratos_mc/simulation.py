import numpy as np

class Simulation:

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed) #allows for repeatability / reproducaility

    def triangular_sim(self, low, mode, high):
        return self.rng.triangular(low, mode, high)

    def pert_sim(self, low, mode, high, lamb=4):
        # PERT = a rescaled Beta distribution
        alpha = 1 + lamb * (mode - low) / (high - low)
        beta = 1 + lamb * (high - mode) / (high - low)
        return low + self.rng.beta(alpha, beta) * (high - low)

    def uniform_sim(self, low, high):
        return self.rng.uniform(low, high)

    def lognormal_sim(self, low, mode, high):
        if mode <= 0:
            return self.triangular_sim(low, mode, high)
        mu = np.log(mode)
        sigma = (np.log(high / low) / (2 * 1.645) if low > 0
                 else np.log(high / mode) / 1.645)
        return self.rng.lognormal(mu, sigma)

    def bernoulli_sim(self, probability):
        """True with the given probability (used for 'does the risk trigger?')."""
        return self.rng.random() < probability

    def three_point(self, low, mode, high, distribution="PERT"):
        """Sample from a low / most likely / high estimate, as written in the spreadsheet."""
        if high <= low:                   # no range (e.g. 0/0/0 delay on a cost-only risk)
            return float(mode)
        d = distribution.strip().lower()
        if d == "triangular":
            return self.triangular_sim(low, mode, high)
        if d == "pert":
            return self.pert_sim(low, mode, high)
        if d == "uniform":
            return self.uniform_sim(low, high)
        if d == "lognormal":
            return self.lognormal_sim(low, mode, high)
        raise ValueError(f"Unknown distribution: {distribution}")

