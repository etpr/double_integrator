import matplotlib.pyplot as plt
import numpy as np
from double_integrator import DoubleIntegrator
from random import random, seed
from tests.tests_configuration import *


def test_simple():
    x_0 = np.array([0.5, -1.0])
    x_f = np.array([-0.5 , 1.0])

    di = DoubleIntegrator(dt=0.001, u_sys_lim=1.0)

    t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
    t_f = t_f_opt + 1.5
    res, traj = di.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)

    if res:
        assert np.linalg.norm(x_0 - traj.X[0]) < 1e-3
        assert np.linalg.norm(x_f - traj.X[-1]) < 1e-2


def test_final_state():
    success_count = 0
    for i in range(N_RAND):
        np.random.seed(i)
        x_0 = np.random.rand(2) * 2.0 - 1.0
        x_f = np.random.rand(2) * 2.0 - 1.0
        u_lim = random() * 5.0 + 0.5

        di = DoubleIntegrator(dt=0.001, u_sys_lim=u_lim)
        t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
        t_f = t_f_opt * 3.0

        res, traj = di.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)
        if res:
            success_count += 1
            assert np.linalg.norm(x_f - traj.X[-1]) < 1e-2

    print('energy optimal control success rate : ', success_count, ' / ', N_RAND)


def test_bound_states():
    for i in range(N_RAND):
        np.random.seed(i)
        x_0 = np.random.rand(2) * 2.0 - 1.0
        x_f = np.copy(x_0)
        u_lim = random() * 5.0 + 0.5

        di = DoubleIntegrator(dt=0.001, u_sys_lim=u_lim)
        t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
        t_f = t_f_opt + 1.0

        res, traj = di.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)

        if res:
            assert np.linalg.norm(x_f - traj.X[-1]) < 1e-2
