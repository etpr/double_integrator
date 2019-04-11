import matplotlib.pyplot as plt
import numpy as np
from double_integrator import DoubleIntegrator
from random import random, seed

N_RAND = 100
VIS_TEST = False


def test_visualization():
    x_0 = np.array([0.5, -1.0])
    x_f = np.array([0.0, 0.0])

    di = DoubleIntegrator()
    res, traj = di.time_opt_control(x_0=x_0, x_f=x_f)

    if res:
        plt.figure(1)
        traj.plot_phase()
        plt.figure(2)
        traj.plot_traj()
        if VIS_TEST:
            plt.show()


def test_compare_time_energy_fuel():
    pass
# def test_equal_start_end():
#     x_0 = np.array([0.0, 1.0])
#     x_f = np.array([0.0, 1.0])
#
#     di = DoubleIntegrator()
#     res, traj = di.time_opt_control(x_0=x_0, x_f=x_f)
#
#     assert res
#     assert np.isclose(np.sum(np.abs(traj.S)), 0)
#     assert np.isclose(np.linalg.norm(x_f - traj.X[-1]), 0)
#     assert np.isclose(np.linalg.norm(x_0 - traj.X[0]), 0)
#     assert np.isclose(traj.U[-1], 0)
#     assert traj.t_s == traj.t_f
#
#
# def test_control_limit_from_final_time():
#     for i in range(N_RAND):
#         x_0 = np.random.rand(2) * 2.0 - 1.0
#         x_f = np.random.rand(2) * 2.0 - 1.0
#         u_lim = random() * 5.0 + 0.1
#
#         di = DoubleIntegrator(u_sys_lim=u_lim)
#         res, traj = di.time_opt_control(x_0=x_0, x_f=x_f)
#         assert np.isclose(di.control_limit_from_final_time(x_0=x_0, x_f=x_f, t_f=traj.t_f, s_0=traj.S[0]), u_lim)
