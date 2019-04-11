import matplotlib.pyplot as plt
import numpy as np
from double_integrator_2d import DoubleIntegrator2D
from random import random, seed

N_RAND = 100


def test_simple():
    x_0 = np.array([0.5, -1.0])
    x_f = np.array([0.0, 0.0])

    y_0 = np.array([0.5, -1.0])
    y_f = np.array([0.0, 0.0])

    di = DoubleIntegrator2D()
    res, traj_x, traj_y = di.steer(x_0=x_0, x_f=x_f, y_0=y_0, y_f=y_f)

    assert res
    assert np.isclose(np.linalg.norm(x_f - traj_x.X[-1]), 0)
    assert np.isclose(np.linalg.norm(y_f - traj_y.X[-1]), 0)
    assert np.isclose(np.linalg.norm(x_0 - traj_x.X[0]), 0)
    assert np.isclose(np.linalg.norm(y_0 - traj_y.X[0]), 0)


def test_final_state():
    success = 0
    for i in range(N_RAND):
        np.random.seed(i)
        x_0 = np.random.rand(2) * 2.0 - 1.0
        x_f = np.random.rand(2) * 2.0 - 1.0
        y_0 = np.random.rand(2) * 2.0 - 1.0
        y_f = np.random.rand(2) * 2.0 - 1.0
        u_lim = random() * 5.0 + 0.5
        u_lim = 1.0
        di = DoubleIntegrator2D(dt=0.001, u_sys_lim=u_lim)
        res, traj_x, traj_y = di.steer(x_0=x_0, x_f=x_f, y_0=y_0, y_f=y_f)

        if res:
            success += 1
            assert np.linalg.norm(x_f - traj_x.X[-1]) < 1e-2
            assert np.linalg.norm(y_f - traj_y.X[-1]) < 1e-2
            assert np.linalg.norm(x_0 - traj_x.X[0]) < 1e-2
            assert np.linalg.norm(y_0 - traj_y.X[0]) < 1e-2

    print(success, ' successful trials from ', N_RAND)
#
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
