import matplotlib.pyplot as plt
import numpy as np
from double_integrator import DoubleIntegrator
from random import random, seed

N_RAND = 10


def test_simple():
    x_0 = np.array([0.6, 0.6])
    x_f = np.array([0.0, 0.0])

    di = DoubleIntegrator(dt=0.001, u_sys_lim=1.0)

    t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
    t_f = t_f_opt + 1.0
    res, traj = di.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)

    assert traj.t_s[0] < traj.t_s[1]
    assert traj.t_s[1] < traj.t_f
    assert np.linalg.norm(x_0 - traj.X[0]) < 1e-3
    assert np.linalg.norm(x_f - traj.X[-1]) < 1e-3


def test_bound():
    x_0 = np.array([0.5, -1.0])
    x_f = np.array([0.0, 0.0])
    di = DoubleIntegrator(dt=0.001, u_sys_lim=1.0)
    t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
    t_f = t_f_opt + 1.0
    res, traj = di.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)
    assert traj.t_s[0] <= traj.t_s[1]
    assert traj.t_s[1] <= traj.t_f
    assert np.linalg.norm(x_0 - traj.X[0]) < 1e-3
    assert np.linalg.norm(x_f - traj.X[-1]) < 1e-3

    x_0 = np.array([-0.5, 1.0])
    x_f = np.array([0.0, 0.0])
    di = DoubleIntegrator(dt=0.001, u_sys_lim=1.0)
    t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
    t_f = t_f_opt + 1.0
    res, traj = di.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)
    assert traj.t_s[0] <= traj.t_s[1]
    assert traj.t_s[1] <= traj.t_f
    assert np.linalg.norm(x_0 - traj.X[0]) < 1e-3
    assert np.linalg.norm(x_f - traj.X[-1]) < 1e-3


def test_final_state():
    success = 0
    for i in range(N_RAND):
        x_0 = np.random.rand(2) * 2.0 - 1.0
        x_f = np.random.rand(2) * 2.0 - 1.0
        u_lim = 1.0

        di = DoubleIntegrator(dt=0.001, u_sys_lim=u_lim)

        t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
        t_f = t_f_opt * 1.3
        res, traj = di.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)

        if res:
            success += 1
            assert np.linalg.norm(x_f - traj.X[-1]) < 1e-2
            assert traj.t_s[0] < traj.t_s[1]
            assert traj.t_s[1] < traj.t_f

    print(success, ' successful trials from ', N_RAND)

# def test_final_state_bku():
#     x_0 = np.random.rand(2) * 2.0 - 1.0
#     x_f = np.random.rand(2) * 2.0 - 1.0
#     u_lim = 1.0
#
#     di = DoubleIntegrator(dt=0.001, u_sys_lim=u_lim)
#
#     t_f_opt = di.time_optimal_solution(x_0, x_f)[1]
#     t_f = t_f_opt * 1.2
#     res, traj = di.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f)
#
#     if res:
#         print(x_0, x_f)
#         assert np.linalg.norm(x_f - traj.X[-1]) < 1e-1
#         assert traj.t_s[0] < traj.t_s[1]
#         assert traj.t_s[1] < traj.t_f

# def test_boundary():
#     pass

# def test_final_state():
#     for i in range(N_RAND):
#         x_0 = np.random.rand(2) * 2.0 - 1.0
#         x_f = np.random.rand(2) * 2.0 - 1.0
#         u_lim = random() * 5.0 + 0.5
#
#         di = DoubleIntegrator(dt=0.001, u_sys_lim=u_lim)
#         res, traj = di.time_opt_control(x_0=x_0, x_f=x_f)
#         assert np.linalg.norm(x_f - traj.X[-1]) < 1e-2
#         assert traj.t_s < traj.t_f
#
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
