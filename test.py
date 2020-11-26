import numpy as np
import pandas as pd
from autoportfolio.optimize import minimum_volatility, minimum_tracking_error

def mvp_test_0():
    minimum_volatility(np.zeros((5,5)))

def mvp_test_1():
    cov_mat = np.array([
        [0.000245, 0.000084, 0.000122, 0.000142],
        [0.000084, 0.000219, 0.000085, 0.000092],
        [0.000122, 0.000085, 0.000221, 0.000176],
        [0.000142, 0.000092, 0.000176, 0.000333]
    ])
    minimum_volatility(cov_mat)
    minimum_volatility(cov_mat, convex=False)

def mvp_test_2():
    cov_mat = np.array([
        [0.000245, 0.000084, 0.000122, 0.000142],
        [0.000084, 0.000219, 0.000085, 0.000092],
        [0.000122, 0.000085, 0.000221, 0.000176],
        [0.000142, 0.000092, 0.000176, 0.000333]
    ])
    minimum_volatility(cov_mat, w_lower=0, w_upper=.15)

def mvp_test_3():
    n = 100
    cov_mat = np.random.rand(n,n)
    minimum_volatility(cov_mat)

def mvp_test_4():
    cov_mat = np.eye(50)
    minimum_volatility(cov_mat)

def mvp_test_5():
    cov_mat = np.random.rand(50,50)
    cov_mat[0,:] = 0
    cov_mat[:,0] = 0
    minimum_volatility(cov_mat, convex=False)



def mtep_test_0():
    minimum_tracking_error(np.zeros((5,5)))

def mtep_test_1():
    cov_mat = np.array([
        [0.000245, 0.000084, 0.000122, 0.000142],
        [0.000084, 0.000219, 0.000085, 0.000092],
        [0.000122, 0.000085, 0.000221, 0.000176],
        [0.000142, 0.000092, 0.000176, 0.000333]
    ])
    minimum_tracking_error(cov_mat)
    minimum_tracking_error(cov_mat, convex=False)

def mtep_test_2():
    cov_mat = np.array([
        [0.000245, 0.000084, 0.000122, 0.000142],
        [0.000084, 0.000219, 0.000085, 0.000092],
        [0.000122, 0.000085, 0.000221, 0.000176],
        [0.000142, 0.000092, 0.000176, 0.000333]
    ])
    minimum_tracking_error(cov_mat, w_lower=0, w_upper=.15)

def mtep_test_3():
    n = 100
    cov_mat = np.random.rand(n,n)
    minimum_tracking_error(cov_mat)

def mtep_test_4():
    cov_mat = np.eye(50)
    minimum_tracking_error(cov_mat)

def mtep_test_5():
    cov_mat = np.random.rand(50,50)
    cov_mat[0,:] = 0
    cov_mat[:,0] = 0
    minimum_tracking_error(cov_mat, convex=False)


if __name__ == "__main__":
    mvp_test_0()
    mvp_test_1()
    mvp_test_2()
    mvp_test_3()
    mvp_test_4()
    mvp_test_5()

    mtep_test_0()
    mtep_test_1()
    mtep_test_2()
    mtep_test_3()
    mtep_test_4()
    mtep_test_5()