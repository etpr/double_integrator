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

    def time_optimal_solution(self, x_10, x_20, x_1f, x_2f, u_lim=None):
        if u_lim is not None:
            if u_lim > self.u_sys_lim:
                raise Exception("the provided control limit is above the system limit")
            u_max = u_lim
            u_min = -u_lim
        else:
            u_max = self.u_sys_lim
            u_min = -self.u_sys_lim

        s0 = self.switch_criteria(x_10, x_20, x_1f, x_2f, u_max)
        if s0 > 0:
            t_s = (x_20 + np.sqrt(0.5 * x_20 * x_20 + u_max * x_10 + 0.5 * x_2f * x_2f - u_max * x_1f)) / u_max
            t_f = (x_20 + x_2f + 2 * np.sqrt(
                0.5 * x_20 * x_20 + u_max * x_10 + 0.5 * x_2f * x_2f - u_max * x_1f)) / u_max
            x_1s = 0.5 * t_s * t_s * u_min + x_20 * t_s + x_10
            x_2s = u_min * t_s + x_20
            u_opt_1 = u_min
            u_opt_2 = u_max
        elif s0 < 0:
            t_s = (-x_20 + np.sqrt(0.5 * x_20 * x_20 - u_max * x_10 + 0.5 * x_2f * x_2f + u_max * x_1f)) / u_max
            t_f = (-x_20 + -x_2f + 2 * np.sqrt(
                0.5 * x_20 * x_20 - u_max * x_10 + 0.5 * x_2f * x_2f + u_max * x_1f)) / u_max
            x_1s = 0.5 * t_s * t_s * u_max + x_20 * t_s + x_10
            x_2s = u_max * t_s + x_20
            u_opt_1 = u_max
            u_opt_2 = u_min
        else:
            t_s = 0
            t_f = (np.abs(x_20) + np.abs(x_2f)) / u_max
            x_1s = x_10
            x_2s = x_20
            u_opt_1 = -np.sign(x_20) * u_max
            u_opt_2 = -np.sign(x_20) * u_max

        return s0, t_f, t_s, x_1s, x_2s, u_opt_1, u_opt_2, u_max

    def time_opt_control(self, x_10, x_20, x_1f, x_2f, u_lim=None):
        if x_10 == x_1f and x_20 == x_2f:
            return 0, 0, x_10, x_20, np.array([[x_10, x_20]]), np.array([0]), np.array([0]), np.array([0])

        s0, t_f, t_s, x_1s, x_2s, u_opt_1, u_opt_2, u_max = self.time_optimal_solution(x_10, x_20, x_1f, x_2f, u_lim)

        t = 0
        x_1 = x_10
        x_2 = x_20
        u = u_opt_1

        X = np.array([[x_1, x_2]])
        U = np.array([u])
        S = np.array([s0])
        T = np.array([t])

        while t <= t_f:
            if t >= t_s:
                u = u_opt_2
                t_d = t - t_s
                x_2 = u * t_d + x_2s
                x_1 = 0.5 * t_d * t_d * u + x_2s * t_d + x_1s
            else:
                u = u_opt_1
                x_2 = u * t + x_20
                x_1 = 0.5 * t * t * u + x_20 * t + x_10

            s = self.switch_criteria(x_1, x_2, x_1f, x_2f, u_max)

            X = np.append(X, np.array([[x_1, x_2]]), axis=0)
            U = np.append(U, u)
            S = np.append(S, s)
            T = np.append(T, t)

            t += self.dt

        return t_f, t_s, x_1s, x_2s, X, U, S, T

    def control_limit_from_final_time(self, x_10, x_20, x_1f, x_2f, t_f, s_0):
        if s_0 >= 0:
            u_lim = (np.sqrt(2) * np.sqrt(2 * (x_10 - x_1f) * (x_10 - x_1f) + 2 * (x_10 - x_1f) * x_20 * t_f + 2 * (
                    x_10 - x_1f) * x_2f * t_f + x_20 * x_20 * t_f * t_f + x_2f * x_2f * t_f * t_f) + 2 * (
                             x_10 - x_1f) + x_20 * t_f + x_2f * t_f) / (t_f * t_f)
        else:
            u_lim = (np.sqrt(2) * np.sqrt(2 * (x_10 - x_1f) * (x_10 - x_1f) + 2 * (x_10 - x_1f) * x_20 * t_f + 2 * (
                    x_10 - x_1f) * x_2f * t_f + x_20 * x_20 * t_f * t_f + x_2f * x_2f * t_f * t_f) - 2 * (
                             x_10 - x_1f) - x_20 * t_f - x_2f * t_f) / (t_f * t_f)

        return u_lim

    def energy_optimal_solution(self, x_10, x_20, x_1f, x_2f, t_f, u_lim=None):
        if u_lim is not None:
            if u_lim > self.u_sys_lim:
                raise Exception("the provided control limit is above the system limit")
            u_max = u_lim
            u_min = -u_lim
        else:
            u_max = self.u_sys_lim
            u_min = -self.u_sys_lim

        s0 = self.switch_criteria(x_10, x_20, x_1f, x_2f, u_max)
        if s0 > 0:
            a = u_min
            b = -x_2f - t_f * u_min + x_20
            c = 0.5 * u_min * (x_2f + t_f * u_min - x_20) * (x_2f + t_f * u_min - x_20) - t_f * (
                x_2f + t_f * u_min) + 0.5 * t_f * t_f * u_min - x_10 + x_1f

            t_s2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
            t_s1 = (x_2f + t_f * u_min - x_20 - t_s2 * u_min) / u_min
            x_1s1 = 0.5 * t_s1 * t_s1 * u_min + x_20 * t_s1 + x_10
            x_2s1 = u_min * t_s1 + x_20
            x_1s2 = x_1s1 + (t_s2 - t_s1) * x_2s1
            x_2s2 = x_2s1

            u_opt_1 = u_min
            u_opt_2 = 0
            u_opt_3 = u_max
        elif s0 < 0:
            a = u_max
            b = -x_2f - t_f * u_max + x_20
            c = 0.5 * u_max * (x_2f + t_f * u_max - x_20) * (x_2f + t_f * u_max - x_20) - t_f * (
                x_2f + t_f * u_max) + 0.5 * t_f * t_f * u_max - x_10 + x_1f

            t_s2 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
            t_s1 = (x_2f + t_f * u_max - x_20 - t_s2 * u_max) / u_max
            x_1s1 = 0.5 * t_s1 * t_s1 * u_max + x_20 * t_s1 + x_10
            x_2s1 = u_max * t_s1 + x_20
            x_1s2 = x_1s1 + (t_s2 - t_s1) * x_2s1
            x_2s2 = x_2s1

            u_opt_1 = u_max
            u_opt_2 = 0
            u_opt_3 = u_min
        else:
            raise Exception("This case is not implemented yet")

        return s0, t_f, t_s1, t_s2, x_1s1, x_2s1, x_1s2, x_2s2, u_opt_1, u_opt_2, u_opt_3, u_max

    def energy_opt_control(self, x_10, x_20, x_1f, x_2f, t_f, u_lim=None):
            if x_10 == x_1f and x_20 == x_2f:
                return 0, 0, 0, x_10, x_20, x_10, x_20, np.array([[x_10, x_20]]), np.array([0]), np.array([0]), np.array([0])

            s0, t_f, t_s1, t_s2, x_1s1, x_2s1, x_1s2, x_2s2, u_opt_1, u_opt_2, u_opt_3, u_max = \
                self.energy_optimal_solution(x_10, x_20, x_1f, x_2f, t_f, u_lim)

            # c1 and c2 are parameters of the switching function s(t) = -c1 t + c2
            if s0 > 0:
                c1 = -2.0 / (t_s1 - t_s2)
                c2 = 1 + c1 * t_s1
            else:
                c1 = 2.0 / (t_s1 - t_s2)
                c2 = -1 + c1 * t_s1

            t = 0
            s0 = c2
            x_1 = x_10
            x_2 = x_20
            u = u_opt_1

            X = np.array([[x_1, x_2]])
            U = np.array([u])
            S = np.array([s0])
            T = np.array([t])

            while t <= t_f:
                if t < t_s1:
                    u = u_opt_1
                    x_2 = u * t + x_20
                    x_1 = 0.5 * t * t * u + x_20 * t + x_10
                elif t < t_s2:
                    u = u_opt_2
                    t_d = t - t_s1
                    x_2 = u * t_d + x_2s1
                    x_1 = 0.5 * t_d * t_d * u + x_2s1 * t_d + x_1s1
                else:
                    u = u_opt_3
                    t_d = t - t_s2
                    x_2 = u * t_d + x_2s2
                    x_1 = 0.5 * t_d * t_d * u + x_2s2 * t_d + x_1s2

                s = -c1 * t + c2

                X = np.append(X, np.array([[x_1, x_2]]), axis=0)
                U = np.append(U, u)
                S = np.append(S, s)
                T = np.append(T, t)

                t += self.dt

            return t_f, t_s1, t_s2, x_1s1, x_2s1, x_1s2, x_2s2, X, U, S, T

    def draw(self, X, U):
        import matplotlib.pyplot as plt
        plot_idx = np.bitwise_and(np.bitwise_not(np.isclose(U, 0)), U > 0)
        plt.plot(X[plot_idx, 0], X[plot_idx, 1], 'b')
        plot_idx = np.bitwise_and(np.bitwise_not(np.isclose(U, 0)), U < 0)
        plt.plot(X[plot_idx, 0], X[plot_idx, 1], 'r')
        plt.plot(X[np.isclose(U, 0), 0], X[np.isclose(U, 0), 1], 'k')

    def animate(self, X, U, T):
        import matplotlib.pyplot as plt
        for x, u, t in zip(X, U, T):
            if np.isclose(u, 0):
                plt.plot(x[0], x[1], 'k.')
            elif u > 0:
                plt.plot(x[0], x[1], 'b.')
            else:
                plt.plot(x[0], x[1], 'r.')

            plt.draw()
            plt.pause(1e-12)
