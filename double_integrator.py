import numpy as np


class DoubleIntegrator:
    def __init__(self, u_sys_lim=1, dt=0.1):
        self.u_sys_lim = u_sys_lim
        self.dt = dt

    def switch_criteria(self, x_1, x_2, x_1f, x_2f, u_lim):
        if x_2 - x_2f == 0:
            return (x_2 * x_2 - x_2f * x_2f) / (2 * u_lim) - x_1f + x_1
        else:
            return (x_2 * x_2 - x_2f * x_2f) / (2 * np.sign(x_2 - x_2f) * u_lim) - x_1f + x_1

    def time_opt_control(self, x_10, x_20, x_1f, x_2f, u_lim=None):
        if u_lim is not None:
            if u_lim > self.u_sys_lim:
                raise Exception("the provided control limit is above the system limit")
            u_max = u_lim
            u_min = -u_lim
        else:
            u_max = self.u_sys_lim
            u_min = -self.u_sys_lim

        if x_10 == x_1f and x_20 == x_2f:
            return 0, 0, x_10, x_20, np.array([[x_10, x_20]]), np.array([0]), np.array([0]), np.array([0])

        s0 = self.switch_criteria(x_10, x_20, x_1f, x_2f, u_max)

        # compute final time, switching time and switching state
        if s0 > 0:
            t_s = (x_20 + np.sqrt(0.5 * x_20 * x_20 + u_max * x_10 + 0.5 * x_2f * x_2f - u_max * x_1f)) / u_max
            t_f = (x_20 + x_2f + 2 * np.sqrt(
                0.5 * x_20 * x_20 + u_max * x_10 + 0.5 * x_2f * x_2f - u_max * x_1f)) / u_max
            x_1s = 0.5 * t_s * t_s * u_min + x_20 * t_s + x_10
            x_2s = u_min * t_s + x_20
            u_1 = u_min
            u_2 = u_max
        elif s0 < 0:
            t_s = (-x_20 + np.sqrt(0.5 * x_20 * x_20 - u_max * x_10 + 0.5 * x_2f * x_2f + u_max * x_1f)) / u_max
            t_f = (-x_20 + -x_2f + 2 * np.sqrt(
                0.5 * x_20 * x_20 - u_max * x_10 + 0.5 * x_2f * x_2f + u_max * x_1f)) / u_max
            x_1s = 0.5 * t_s * t_s * u_max + x_20 * t_s + x_10
            x_2s = u_max * t_s + x_20
            u_1 = u_max
            u_2 = u_min
        else:
            t_s = 0
            t_f = (np.abs(x_20) + np.abs(x_2f)) / u_max
            x_1s = x_10
            x_2s = x_20
            u_1 = -np.sign(x_20) * u_max
            u_2 = -np.sign(x_20) * u_max

        # compute optimal state and action trajectories
        t = 0
        x_1 = x_10
        x_2 = x_20
        u = u_1

        X = np.array([[x_1, x_2]])
        U = np.array([u])
        S = np.array([s0])
        T = np.array([t])

        while t <= t_f:
            if t >= t_s:
                u = u_2
                t_d = t - t_s
                x_2 = u * t_d + x_2s
                x_1 = 0.5 * t_d * t_d * u + x_2s * t_d + x_1s
            else:
                u = u_1
                x_2 = u * t + x_20
                x_1 = 0.5 * t * t * u + x_20 * t + x_10

            s = self.switch_criteria(x_1, x_2, x_1f, x_2f, u_max)

            X = np.append(X, np.array([[x_1, x_2]]), axis=0)
            U = np.append(U, u)
            S = np.append(S, s)
            T = np.append(T, t)

            t += self.dt

        return t_f, t_s, x_1s, x_2s, X, U, S, T

    def draw(self, X, S):
        import matplotlib.pyplot as plt
        plot_idx = np.bitwise_and(np.bitwise_not(np.isclose(S, 0)), S > 0)
        plt.plot(X[plot_idx, 0], X[plot_idx, 1], 'r')
        plot_idx = np.bitwise_and(np.bitwise_not(np.isclose(S, 0)), S < 0)
        plt.plot(X[plot_idx, 0], X[plot_idx, 1], 'b')
        plt.plot(X[np.isclose(S, 0), 0], X[np.isclose(S, 0), 1], 'k')
