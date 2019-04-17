import numpy as np
from double_integrator import DoubleIntegrator, Trajectory


class DoubleIntegrator2D:
    def __init__(self, u_sys_lim=1, dt=0.1):
        self.di_x = DoubleIntegrator(u_sys_lim, dt)
        self.di_y = DoubleIntegrator(u_sys_lim, dt)

    def time_opt_metric(self, x_0, x_f, y_0, y_f):
        t_f_x = self.di_x.time_optimal_solution(x_0=x_0, x_f=x_f)[1]
        t_f_y = self.di_y.time_optimal_solution(x_0=y_0, x_f=y_f)[1]

        return np.max([t_f_x, t_f_y])

    def steer_energy_opt(self, x_0, x_f, y_0, y_f):
        s_x0, t_f_x = self.di_x.time_optimal_solution(x_0=x_0, x_f=x_f)[0:2]
        s_y0, t_f_y = self.di_y.time_optimal_solution(x_0=y_0, x_f=y_f)[0:2]

        t_f = np.max(np.array([t_f_x, t_f_y]))
        t_f_min = t_f
        t_f_max = t_f * 5.0
        res_x = self.di_x.energy_optimal_solution(x_0=x_0, x_f=x_f, t_f=t_f_max)[0]
        res_y = self.di_y.energy_optimal_solution(x_0=y_0, x_f=y_f, t_f=t_f_max)[0]

        if not (res_x and res_y):
            return False, None, None

        # binary search to find shortest feasible duration
        n_iter = 10
        for i in range(n_iter):
            t_f_i = (t_f_max + t_f_min) / 2.0
            res_x = self.di_x.energy_optimal_solution(x_0=x_0, x_f=x_f, t_f=t_f_i)[0]
            res_y = self.di_y.energy_optimal_solution(x_0=y_0, x_f=y_f, t_f=t_f_i)[0]

            if res_x and res_y:
                t_f_max = t_f_i
            else:
                t_f_min = t_f_i

        t_f_opt = t_f_max
        res_x, traj_x = self.di_x.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f_opt)
        res_y, traj_y = self.di_y.energy_opt_control(x_0=y_0, x_f=y_f, t_f=t_f_opt)

        return True, traj_x, traj_y

    def steer_time_opt(self, x_0, x_f, y_0, y_f):
        s_x0, t_f_x = self.di_x.time_optimal_solution(x_0=x_0, x_f=x_f)[0:2]
        s_y0, t_f_y = self.di_y.time_optimal_solution(x_0=y_0, x_f=y_f)[0:2]

        if np.isclose(t_f_x, 0) and np.isclose(t_f_y, 0):  # trivial case: start==goal
            traj_x = Trajectory(t_f=0, t_s=0, x_s=x_0, x_0=x_0, x_f=x_f, X=np.array([y_f]), U=np.zeros(1), S=np.zeros(1),
                                T=np.zeros(1))
            traj_y = Trajectory(t_f=0, t_s=0, x_s=y_0, x_0=y_0, x_f=y_f, X=np.array([y_f]), U=np.zeros(1), S=np.zeros(1),
                                T=np.zeros(1))
            return True, traj_x, traj_y
        elif np.isclose(t_f_x, 0):  # movement only along y direction
            res, traj_y = self.di_y.time_opt_control(x_0=y_0, x_f=y_f)
            n = len(traj_y.T)
            X = np.repeat(np.array([x_0]), n, axis=0)
            traj_x = Trajectory(t_f=t_f_y, t_s=0, x_s=y_0, x_0=x_0, x_f=x_f, X=X, U=np.zeros(n), S=np.zeros(n), T=traj_y.T)
            return True, traj_x, traj_y

        elif np.isclose(t_f_y, 0):  # movement only along x direction
            res, traj_x = self.di_x.time_opt_control(x_0=x_0, x_f=x_f)
            n = len(traj_x.T)
            Y = np.repeat(np.array([y_0]), n, axis=0)
            traj_y = Trajectory(t_f=t_f_x, t_s=0, x_s=y_0, x_0=y_0, x_f=y_f, X=Y, U=np.zeros(n), S=np.zeros(n), T=traj_x.T)
            return True, traj_x, traj_y

        elif np.isclose(t_f_x, t_f_y):  # x and y motion take same amount of time
            res, traj_x = self.di_x.time_opt_control(x_0=x_0, x_f=x_f)
            res, traj_y = self.di_y.time_opt_control(x_0=y_0, x_f=y_f)
            return True, traj_x, traj_y

        elif t_f_x < t_f_y:  # x motion is faster than y, adapt x to y's duration
            res, traj_y = self.di_y.time_opt_control(x_0=y_0, x_f=y_f)

            # case 1: bang bang with t_f_x
            u_lim_new = self.di_x.control_limit_from_final_time(x_0=x_0, x_f=x_f, t_f=t_f_y, s_0=s_x0)
            t_f_x = self.di_x.time_optimal_solution(x_0=x_0, x_f=x_f, u_lim=u_lim_new)[1]
            if np.isclose(t_f_x, t_f_y):
                res, traj_x = self.di_x.time_opt_control(x_0=x_0, x_f=x_f, u_lim=u_lim_new)
                return True, traj_x, traj_y

            # case 2: inverted bang bang with t_f_y
            u_lim_new = self.di_x.control_limit_from_final_time(x_0=x_0, x_f=x_f, t_f=t_f_y, s_0=-s_x0)
            if 0 < u_lim_new < self.di_x.u_sys_lim:
                t_f_x = self.di_x.time_optimal_solution(x_0=x_0, x_f=x_f, u_lim=u_lim_new)[1]
                if np.isclose(t_f_x, t_f_y):
                    res, traj_x = self.di_x.time_opt_control(x_0=x_0, x_f=x_f, u_lim=u_lim_new)
                    return True, traj_x, traj_y

            # case 3: fuel optimal control [bang-zero-bang with t_f_y
            res, traj_x = self.di_x.fuel_opt_control(x_0=x_0, x_f=x_f, t_f=t_f_y)
            if res:
                return True, traj_x, traj_y

            # no feasible trajectory found
            return False, None, None

        elif t_f_x > t_f_y:  # y motion is faster than x, adapt y to x's duration
            res, traj_x = self.di_x.time_opt_control(x_0=x_0, x_f=x_f)

            # case 1: bang bang with t_f_x
            u_lim_new = self.di_y.control_limit_from_final_time(x_0=y_0, x_f=y_f, t_f=t_f_x, s_0=s_y0)
            t_f_y = self.di_x.time_optimal_solution(x_0=y_0, x_f=y_f, u_lim=u_lim_new)[1]
            if np.isclose(t_f_x, t_f_y):
                res, traj_y = self.di_y.time_opt_control(x_0=y_0, x_f=y_f, u_lim=u_lim_new)
                return True, traj_x, traj_y

            # case 2: inverted bang bang with t_f_x
            u_lim_new = self.di_y.control_limit_from_final_time(x_0=y_0, x_f=y_f, t_f=t_f_x, s_0=-s_y0)
            if 0 < u_lim_new < self.di_y.u_sys_lim:
                t_f_y = self.di_y.time_optimal_solution(x_0=y_0, x_f=y_f, u_lim=u_lim_new)[1]
                if np.isclose(t_f_x, t_f_y):
                    res, traj_y = self.di_y.time_opt_control(x_0=y_0, x_f=y_f, u_lim=u_lim_new)
                    return True, traj_x, traj_y

            # case 3: fuel optimal control with t_f_x
            res, traj_y = self.di_y.fuel_opt_control(x_0=y_0, x_f=y_f, t_f=t_f_x)
            if res:
                return True, traj_x, traj_y

            # no feasible trajectory found
            return False, None, None

    def plot_xy(self, traj_x, traj_y, resize=True):
        import matplotlib.pyplot as plt
        if resize:
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.grid(True, which='both')
            lim = np.max(np.abs(np.concatenate([traj_x.X[:, 0], traj_y.X[:, 0]]))) * 1.2
            plt.ylim([-lim, lim])
            plt.xlim([-lim, lim])
            plt.xlabel('x')
            plt.ylabel('y')

        plt.plot(traj_x.X[:, 0], traj_y.X[:, 0])
        delta = 0.1
        plt.plot(traj_x.x_0[0], traj_y.x_0[0], '.c')
        plt.plot(traj_x.x_f[0], traj_y.x_f[0], '.g')
        plt.arrow(traj_x.x_0[0], traj_y.x_0[0], delta * traj_x.x_0[1], delta * traj_y.x_0[1], color='c', width = 0.01)
        plt.arrow(traj_x.x_f[0], traj_y.x_f[0], delta * traj_x.x_f[1], delta * traj_y.x_f[1], color='g', width=0.01)

    def plot_traj(self, traj_x, traj_y):
        traj_x.plot_traj()
        traj_y.plot_traj()
