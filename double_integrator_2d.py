import numpy as np
from double_integrator import DoubleIntegrator, Trajectory


class DoubleIntegrator2D:
    def __init__(self, u_sys_lim=1, dt=0.1):
        self.di_x = DoubleIntegrator(u_sys_lim, dt)
        self.di_y = DoubleIntegrator(u_sys_lim, dt)

    def steer(self, x_0, x_f, y_0, y_f):
        s_x0, t_f_x = self.di_x.time_optimal_solution(x_0=x_0, x_f=x_f)[0:2]
        s_y0, t_f_y = self.di_y.time_optimal_solution(x_0=y_0, x_f=y_f)[0:2]

        if np.isclose(t_f_x, 0):  # movement only along y direction
            res, traj_y = self.di_y.time_opt_control(x_0=y_0, x_f=y_f)
            n = len(traj_y.T)
            X = np.repeat(np.array([[x_0]]), n, axis=0)
            res, traj_x = True, Trajectory(t_f=t_f_y, t_s=x_0, x_s=x_0, X=X, U=np.zeros(n), S=np.zeros(n),
                                           T=traj_y.T), traj_y

        elif np.isclose(t_f_y, 0):  # movement only along x direction
            res, traj_x = self.di_x.time_opt_control(x_0=x_0, x_f=x_f)
            n = len(traj_x.T)
            Y = np.repeat(np.array([[y_0]]), n, axis=0)
            res, traj_y = True, traj_x, Trajectory(t_f=t_f_x, t_s=y_0, x_s=y_0, X=Y, U=np.zeros(n), S=np.zeros(n),
                                                   T=traj_x.T)

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
                if not np.isclose(t_f_x, t_f_y):
                    res, traj_x = self.di_x.time_opt_control(x_0=x_0, x_f=x_f, u_lim=u_lim_new)
                    return True, traj_x, traj_y

            # case 3: energy optimal control with t_f_y
            res, traj_x = self.di_x.energy_opt_control(x_0=x_0, x_f=x_f, t_f=t_f_y)
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
                if not np.isclose(t_f_x, t_f_y):
                    res, traj_y = self.di_y.time_opt_control(x_0=y_0, x_f=y_f, u_lim=u_lim_new)
                    return True, traj_x, traj_y

            # case 3: energy optimal control with t_f_x
            res, traj_y = self.di_y.energy_opt_control(x_0=y_0, x_f=y_f, t_f=t_f_x)
            if res:
                return True, traj_x, traj_y

            # no feasible trajectory found
            return False, None, None

    def draw(self):
        pass
