import matplotlib.pyplot as plt
import numpy as np
from double_integrator_2d import DoubleIntegrator2D
from random import random, seed
from tests.tests_configuration import *


def test_simple():
    x_0 = np.array([0.5, -1.0])
    x_f = np.array([0.0, 0.0])

    y_0 = np.array([0.5, -1.0])
    y_f = np.array([0.0, 0.0])

    di = DoubleIntegrator2D(dt=0.001)
    res, traj_x, traj_y = di.steer_energy_opt(x_0=x_0, x_f=x_f, y_0=y_0, y_f=y_f)

    assert res
    assert np.equal(traj_x.t_f, traj_y.t_f)
    assert np.isclose(len(traj_x.T), len(traj_y.T))
    assert np.linalg.norm(x_f - traj_x.X[-1]) < 1e-2
    assert np.linalg.norm(y_f - traj_y.X[-1]) < 1e-2
    assert np.linalg.norm(x_0 - traj_x.X[0]) < 1e-2
    assert np.linalg.norm(y_0 - traj_y.X[0]) < 1e-2


def test_boundary_cases():
    x_0 = np.array([0.5, -1.0])
    x_f = np.array([0.5, -1.0])

    y_0 = np.array([0.5, -1.0])
    y_f = np.array([0.0, 0.0])

    di = DoubleIntegrator2D(dt=0.001)
    res, traj_x, traj_y = di.steer_energy_opt(x_0=x_0, x_f=x_f, y_0=y_0, y_f=y_f)

    assert res
    assert np.isclose(len(traj_x.T), len(traj_y.T))
    assert np.linalg.norm(x_f - traj_x.X[-1]) < 1e-2
    assert np.linalg.norm(y_f - traj_y.X[-1]) < 1e-2
    assert np.linalg.norm(x_0 - traj_x.X[0]) < 1e-2
    assert np.linalg.norm(y_0 - traj_y.X[0]) < 1e-2


def test_final_state_energy_opt():
    success = 0
    for i in range(N_RAND):
        np.random.seed(i)
        x_0 = np.random.rand(2) * 2.0 - 1.0
        x_f = np.random.rand(2) * 2.0 - 1.0
        y_0 = np.random.rand(2) * 2.0 - 1.0
        y_f = np.random.rand(2) * 2.0 - 1.0
        u_lim = random() * 5.0 + 0.5

        di = DoubleIntegrator2D(dt=0.001, u_sys_lim=u_lim)
        res, traj_x, traj_y = di.steer_energy_opt(x_0=x_0, x_f=x_f, y_0=y_0, y_f=y_f)

        if res:
            success += 1
            assert np.isclose(len(traj_x.T), len(traj_y.T))
            assert np.linalg.norm(x_f - traj_x.X[-1]) < 1e-2
            assert np.linalg.norm(y_f - traj_y.X[-1]) < 1e-2
            assert np.linalg.norm(x_0 - traj_x.X[0]) < 1e-2
            assert np.linalg.norm(y_0 - traj_y.X[0]) < 1e-2

    print(success, ' successful trials from ', N_RAND)


def test_final_state_time_opt():
    success = 0
    for i in range(N_RAND):
        np.random.seed(i)
        x_0 = np.random.rand(2) * 2.0 - 1.0
        x_f = np.random.rand(2) * 2.0 - 1.0
        y_0 = np.random.rand(2) * 2.0 - 1.0
        y_f = np.random.rand(2) * 2.0 - 1.0
        u_lim = random() * 5.0 + 0.5

        di = DoubleIntegrator2D(dt=0.001, u_sys_lim=u_lim)
        res, traj_x, traj_y = di.steer_time_opt(x_0=x_0, x_f=x_f, y_0=y_0, y_f=y_f)

        if res:
            success += 1
            assert np.isclose(len(traj_x.T), len(traj_y.T))
            assert np.linalg.norm(x_f - traj_x.X[-1]) < 1e-2
            assert np.linalg.norm(y_f - traj_y.X[-1]) < 1e-2
            assert np.linalg.norm(x_0 - traj_x.X[0]) < 1e-2
            assert np.linalg.norm(y_0 - traj_y.X[0]) < 1e-2

    print(success, ' successful trials from ', N_RAND)


def test_visualization():
    x_0 = np.array([0.0, 0.5])
    x_f = np.array([1.0, 0.0])
    y_0 = np.array([0.0, 0.0])
    y_f = np.array([1.0, 0.5])

    di = DoubleIntegrator2D()
    res, traj_x, traj_y = di.steer_energy_opt(x_0=x_0, x_f=x_f, y_0=y_0, y_f=y_f)

    plt.figure(1)
    di.plot_xy(traj_x, traj_y)

    plt.figure(2)
    di.plot_traj(traj_x, traj_y)

    if VIS_TEST:
        plt.show()
