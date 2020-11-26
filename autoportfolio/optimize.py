import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

def minimum_volatility(cov_mat, w_lower=0.0, w_upper=1.0, convex=True):
    m, n = cov_mat.shape
    assert m == n
    assert w_lower <= w_upper

    if convex:
        w = cp.Variable(m)
        objective = cp.Minimize(cp.quad_form(w, cov_mat))
        constraints = [w >= w_lower, w <= w_upper, sum(w) == 1]
        problem = cp.Problem(objective, constraints)

        problem.solve()
        return w.value

    else:
        cost = lambda w: np.sqrt(np.sum(w.T @ cov_mat @ w))
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w)-1},
        ]
        bounds = np.zeros((m, 2))
        bounds[:, 0] = w_lower
        bounds[:, 1] = w_upper
        initial_guess = np.full(m, (w_upper - w_lower)/2)
        res = minimize(
            cost, 
            initial_guess, 
            method="SLSQP", 
            bounds=bounds,
            constraints=constraints
        )

        return res.x


def minimum_tracking_error(cov_mat, w_lower=0.0, w_upper=1.0, convex=True):
    return