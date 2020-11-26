import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

def minimum_volatility(cov_mat, w_lower=0.0, w_upper=1.0, convex=False):
    """
    determine the portfolio allocations w that minimize volatility

    Args:
        cov_mat (numpy array): the variance covariance matrix
        w_lower (float, optional): lower bound for allocation proportion. Defaults to 0.0.
        w_upper (float, optional): upper bound for allocation proportion. Defaults to 1.0.
        convex (bool, optional): whether the problem is convex or not; convexity allows a 
                                 speedup. Defaults to False.

    Returns:
        (numpy array): the values that produce the minimum volatility portfolio
    """    
    m, n = cov_mat.shape
    assert m == n
    assert w_lower <= w_upper

    if convex:
        w = cp.Variable(m)
        objective = cp.Minimize(cp.sqrt(cp.quad_form(w, cov_mat)))
        constraints = [w >= w_lower, w <= w_upper, sum(w) == 1]
        problem = cp.Problem(objective, constraints)

        problem.solve(qcp=True)
        return w.value

    else:
        cost = lambda w: np.sqrt(w.T @ cov_mat @ w)
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
    m, n = cov_mat.shape
    assert m == n
    assert w_lower <= w_upper

    sigma_bm = cov_mat[m-1][m-1]
    rho_p_bm = cov_mat[:,m-1]
    print(sigma_bm)
    print(rho_p_bm)

    if convex:
        w = cp.Variable(m)
        objective = cp.Minimize(cp.sqrt(cp.quad_form(w, cov_mat)))
        constraints = [w >= w_lower, w <= w_upper, sum(w) == 1]
        problem = cp.Problem(objective, constraints)

        problem.solve(qcp=True)
        return w.value

    else:
        cost = lambda w: np.sqrt(w.T @ cov_mat @ w * np.sum(1 - 2*sigma_bm*rho_p_bm) + sigma_bm**2)
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