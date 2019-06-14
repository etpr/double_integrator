import matplotlib.pyplot as plt
import numpy as np
from double_integrator import DoubleIntegrator
from random import random, seed
from tests.tests_configuration import *


def test_simple():
    x_0 = np.array([0.6, 0.5])
    x_f = np.array([0.1, 0.2])

    di = DoubleIntegrator(dt=0.001, u_sys_lim=1.0)

    t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
    t_f = t_f_opt + 1.0
    res, traj = di.fuel_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)

    assert traj.t_s[0] < traj.t_s[1]
    assert traj.t_s[1] < traj.t_f
    assert np.linalg.norm(x_0 - traj.X[0]) < 1e-3
    assert np.linalg.norm(x_f - traj.X[-1]) < 1e-3


def test_switching_boundary():
    x_0 = np.array([0.5, -1.0])
    x_f = np.array([0.0, 0.0])
    di = DoubleIntegrator(dt=0.001, u_sys_lim=1.0)
    t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
    t_f = t_f_opt + 1.0
    res, traj = di.fuel_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)

    assert traj.t_s[0] <= traj.t_s[1]
    assert traj.t_s[1] <= traj.t_f
    assert np.linalg.norm(x_0 - traj.X[0]) < 1e-3
    assert np.linalg.norm(x_f - traj.X[-1]) < 1e-3

    x_0 = np.array([-0.5, 1.0])
    x_f = np.array([0.0, 0.0])
    di = DoubleIntegrator(dt=0.001, u_sys_lim=1.0)
    t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
    t_f = t_f_opt + 1.0
    res, traj = di.fuel_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)
    assert traj.t_s[0] <= traj.t_s[1]
    assert traj.t_s[1] <= traj.t_f
    assert np.linalg.norm(x_0 - traj.X[0]) < 1e-3
    assert np.linalg.norm(x_f - traj.X[-1]) < 1e-3


def test_final_state():
    success_count = 0
    for i in range(N_RAND):
        np.random.seed(i)
        x_0 = np.random.rand(2) * 2.0 - 1.0
        x_f = np.random.rand(2) * 2.0 - 1.0
        u_lim = 1.0

        di = DoubleIntegrator(dt=0.001, u_sys_lim=u_lim)

        t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
        t_f = t_f_opt * 1.05
        res, traj = di.fuel_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)

        if res:
            success_count += 1
            assert np.linalg.norm(x_f - traj.X[-1]) < 1e-2
            assert traj.t_s[0] <= traj.t_s[1]
            assert traj.t_s[1] <= traj.t_f

    print(' fuel optimal control success rate ', success_count, ' / ', N_RAND)


def test_equal_start_end():
    x_0 = np.array([0.0, 1.0])
    x_f = np.array([0.0, 1.0])

    di = DoubleIntegrator()

    t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
    t_f = t_f_opt * (random() + 1)
    res, traj = di.fuel_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)

    assert res
    assert np.isclose(np.sum(np.abs(traj.S)), 0)
    assert np.isclose(np.linalg.norm(x_f - traj.X[-1]), 0)
    assert np.isclose(np.linalg.norm(x_0 - traj.X[0]), 0)
    assert np.isclose(traj.U[-1], 0)
    assert traj.t_s == traj.t_f
