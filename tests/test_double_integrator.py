import matplotlib.pyplot as plt
import numpy as np
from double_integrator import DoubleIntegrator
from tests_configuration import *


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
    x_0 = np.array([0.6, 0.6])
    x_f = np.array([0.0, 0.0])
    di = DoubleIntegrator(dt=0.001, u_sys_lim=1.0)
    t_f_opt = di.time_optimal_solution(x_0, x_f)[1]

    res_to, traj_to = di.time_opt_control(x_0=x_0, x_f=x_f)
    res_eo, traj_eo = di.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f_opt + 2.0)
    res_fo, traj_fo = di.fuel_opt_control(x_0=x_0, x_f=x_f, t_f=t_f_opt + 1.0)

    if res_to and res_fo and res_eo:
        plt.figure(1)
        traj_to.plot_phase()
        traj_eo.plot_phase()
        traj_fo.plot_phase()

        plt.figure(2)
        traj_to.plot_traj()
        traj_eo.plot_traj()
        traj_fo.plot_traj()
        if VIS_TEST:
            plt.show()
