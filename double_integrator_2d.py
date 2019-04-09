import numpy as np
from double_integrator import DoubleIntegrator


class DoubleIntegrator2D:
    def __init__(self, u_sys_lim=1, dt=0.1):
        self.di_x = DoubleIntegrator(u_sys_lim, dt)
        self.di_y = DoubleIntegrator(u_sys_lim, dt)

    def steer(self, x_10, x_20, x_1f, x_2f, y_10, y_20, y_1f, y_2f):
        s_x0, t_f_x = self.di_x.optimal_controller(x_10=x_10, x_20=x_20, x_1f=x_1f, x_2f=x_2f)[0:2]
        s_y0, t_f_y = self.di_y.optimal_controller(x_10=y_10, x_20=y_20, x_1f=y_1f, x_2f=y_2f)[0:2]

        if np.isclose(t_f_x, 0):
            t_f_y, t_s_y, y_1s, y_2s, Y, U_y, S_y, T_y = self.di_y.time_opt_control(x_10=y_10, x_20=y_20, x_1f=y_1f,
                                                                                    x_2f=y_2f)
            n = len(T_y)
            X = np.repeat(np.array([[x_10, x_20]]), n, axis=0)
            t_f_x, t_s_x, y_1s, y_2s, X, U_x, S_x, T_x = t_f_y, 0, x_10, x_20, X, np.zeros(n), np.zeros(n), T_y
        elif np.isclose(t_f_y, 0):
            t_f_x, t_s_x, y_1s, y_2s, X, U_x, S_x, T_x = self.di_x.time_opt_control(x_10=x_10, x_20=x_20, x_1f=x_1f,
                                                                                    x_2f=x_2f)
            n = len(T_x)
            Y = np.repeat(np.array([[y_10, y_20]]), n, axis=0)
            t_f_y, t_s_y, y_1s, y_2s, Y, U_y, S_y, T_y = t_f_x, 0, y_10, y_20, Y, np.zeros(n), np.zeros(n), T_x
        elif np.isclose(t_f_x, t_f_y):
            t_f_x, t_s_x, y_1s, y_2s, X, U_x, S_x, T_x = self.di_x.time_opt_control(x_10=x_10, x_20=x_20, x_1f=x_1f,
                                                                                    x_2f=x_2f)
            t_f_y, t_s_y, y_1s, y_2s, Y, U_y, S_y, T_y = self.di_y.time_opt_control(x_10=y_10, x_20=y_20, x_1f=y_1f,
                                                                                    x_2f=y_2f)
        elif t_f_x < t_f_y:
            u_lim_new = self.di_x.comp_limit_from_final_time(x_10=x_10, x_20=x_20, x_1f=x_1f, x_2f=x_2f, t_f=t_f_y,
                                                             s_0=s_x0)
            t_f_x = self.di_x.optimal_controller(x_10=x_10, x_20=x_20, x_1f=x_1f, x_2f=x_2f, u_lim=u_lim_new)[1]

            if not np.isclose(t_f_x, t_f_y):
                u_lim_new = self.di_x.comp_limit_from_final_time(x_10=x_10, x_20=x_20, x_1f=x_1f, x_2f=x_2f, t_f=t_f_y,
                                                                 s_0=-s_x0)
                if u_lim_new > 0 and u_lim_new < self.di_x.u_sys_lim:
                    t_f_x = self.di_x.optimal_controller(x_10=x_10, x_20=x_20, x_1f=x_1f, x_2f=x_2f, u_lim=u_lim_new)[1]
                if not np.isclose(t_f_x, t_f_y):
                    return False, []

            t_f_x, t_s_x, y_1s, y_2s, X, U_x, S_x, T_x = self.di_x.time_opt_control(x_10=x_10, x_20=x_20, x_1f=x_1f,
                                                                                    x_2f=x_2f, u_lim=u_lim_new)
            t_f_y, t_s_y, y_1s, y_2s, Y, U_y, S_y, T_y = self.di_y.time_opt_control(x_10=y_10, x_20=y_20, x_1f=y_1f,
                                                                                    x_2f=y_2f)
        elif t_f_x > t_f_y:
            u_lim_new = self.di_y.comp_limit_from_final_time(x_10=y_10, x_20=y_20, x_1f=y_1f, x_2f=y_2f, t_f=t_f_x,
                                                             s_0=s_y0)
            t_f_y = self.di_x.optimal_controller(x_10=y_10, x_20=y_20, x_1f=y_1f, x_2f=y_2f, u_lim=u_lim_new)[1]

            if not np.isclose(t_f_x, t_f_y):
                u_lim_new = self.di_y.comp_limit_from_final_time(x_10=y_10, x_20=y_20, x_1f=y_1f, x_2f=y_2f, t_f=t_f_x,
                                                                 s_0=-s_y0)
                if u_lim_new > 0 and u_lim_new < self.di_y.u_sys_lim:
                    t_f_y = self.di_x.optimal_controller(x_10=y_10, x_20=y_20, x_1f=y_1f, x_2f=y_2f, u_lim=u_lim_new)[1]
                if not np.isclose(t_f_x, t_f_y):
                    return False, []

            t_f_y, t_s_y, y_1s, y_2s, Y, U_y, S_y, T_y = self.di_y.time_opt_control(x_10=y_10, x_20=y_20, x_1f=y_1f,
                                                                                    x_2f=y_2f, u_lim=u_lim_new)
            t_f_x, t_s_x, y_1s, y_2s, X, U_x, S_x, T_x = self.di_x.time_opt_control(x_10=x_10, x_20=x_20, x_1f=x_1f,
                                                                                    x_2f=x_2f)

        return True, [t_f_x, t_s_x, y_1s, y_2s, X, U_x, S_x, T_x, t_f_y, t_s_y, y_1s, y_2s, Y, U_y, S_y, T_y]

    def draw(self):
        pass
