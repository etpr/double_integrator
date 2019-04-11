import numpy as np


class DoubleIntegrator:
    def __init__(self, u_sys_lim=1, dt=0.1):
        self.u_sys_lim = u_sys_lim
        self.dt = dt

    def switch_criteria(self, x, x_f, u_lim):
        if x[1] - x_f[1] == 0:
            return (x[1] * x[1] - x_f[1] * x_f[1]) / (2 * u_lim) - x_f[0] + x[0]
        else:
            return (x[1] * x[1] - x_f[1] * x_f[1]) / (2 * np.sign(x[1] - x_f[1]) * u_lim) - x_f[0] + x[0]

    def time_optimal_solution(self, x_0, x_f, u_lim=None):
        if u_lim is not None:
            if u_lim > self.u_sys_lim:
                raise Exception("the provided control limit is above the system limit")
            u_max = u_lim
            u_min = -u_lim
        else:
            u_max = self.u_sys_lim
            u_min = -self.u_sys_lim

        s0 = self.switch_criteria(x_0, x_f, u_max)

        if s0 > 0:
            t_s = (x_0[1] + np.sqrt(0.5 * x_0[1] * x_0[1] + u_max * x_0[0] + 0.5 * x_f[1] * x_f[1] - u_max * x_f[0])) / u_max
            t_f = (x_0[1] + x_f[1] + 2 * np.sqrt(
                0.5 * x_0[1] * x_0[1] + u_max * x_0[0] + 0.5 * x_f[1] * x_f[1] - u_max * x_f[0])) / u_max
            x_s = np.array([0.5 * t_s * t_s * u_min + x_0[1] * t_s + x_0[0], u_min * t_s + x_0[1]])
            u_opt = [u_min, u_max]
        elif s0 < 0:
            t_s = (-x_0[1] + np.sqrt(0.5 * x_0[1] * x_0[1] - u_max * x_0[0] + 0.5 * x_f[1] * x_f[1] + u_max * x_f[0])) / u_max
            t_f = (-x_0[1] + -x_f[1] + 2 * np.sqrt(
                0.5 * x_0[1] * x_0[1] - u_max * x_0[0] + 0.5 * x_f[1] * x_f[1] + u_max * x_f[0])) / u_max
            x_s = np.array([0.5 * t_s * t_s * u_max + x_0[1] * t_s + x_0[0], u_max * t_s + x_0[1]])
            u_opt = [u_max, u_min]
        else:
            t_s = 0
            t_f = (np.abs(x_0[1]) + np.abs(x_f[1])) / u_max
            x_s = np.copy(x_0)
            u_opt = [-np.sign(x_0[1]) * u_max]

        return s0, t_f, t_s, x_s, u_opt, u_max

    def time_opt_control(self, x_0, x_f, u_lim=None):
        if x_0[0] == x_f[0] and x_0[1] == x_f[1]:  # trivial case: start == goal
            return True, Trajectory(t_f=0, t_s=0, x_s=x_0, X=np.array([x_0]), U=np.array([0]), S=np.array([0]),
                                    T=np.array([0]))

        s0, t_f, t_s, x_s, u_opt, u_max = self.time_optimal_solution(x_0, x_f, u_lim)

        t = 0
        x = np.copy(x_0)
        u = u_opt[0]

        X = np.array([x_0])
        U = np.array([u])
        S = np.array([s0])
        T = np.array([t])

        while t <= t_f:
            if t >= t_s:
                u = u_opt[-1]
                t_d = t - t_s
                x[1] = u * t_d + x_s[1]
                x[0] = 0.5 * t_d**2 * u + x_s[1] * t_d + x_s[0]
            else:
                u = u_opt[0]
                x[1] = u * t + x_0[1]
                x[0] = 0.5 * t ** 2 * u + x_0[1] * t + x_0[0]

            s = self.switch_criteria(x, x_f, u_max)

            X = np.append(X, np.array([x]), axis=0)
            U = np.append(U, u)
            S = np.append(S, s)
            T = np.append(T, t)

            t += self.dt

        return True, Trajectory(t_f=t_f, t_s=t_s, x_s=x_s, X=X, U=U, S=S, T=T)

    def control_limit_from_final_time(self, x_0, x_f, t_f, s_0):
        if s_0 >= 0:
            u_lim = (np.sqrt(2) * np.sqrt(
                2 * (x_0[0] - x_f[0]) * (x_0[0] - x_f[0]) + 2 * (x_0[0] - x_f[0]) * x_0[1] * t_f + 2 * (
                    x_0[0] - x_f[0]) * x_f[1] * t_f + x_0[1] * x_0[1] * t_f * t_f + x_f[1] * x_f[1] * t_f * t_f) + 2 * (
                         x_0[0] - x_f[0]) + x_0[1] * t_f + x_f[1] * t_f) / (t_f * t_f)
        else:
            u_lim = (np.sqrt(2) * np.sqrt(
                2 * (x_0[0] - x_f[0]) * (x_0[0] - x_f[0]) + 2 * (x_0[0] - x_f[0]) * x_0[1] * t_f + 2 * (
                    x_0[0] - x_f[0]) * x_f[1] * t_f + x_0[1] * x_0[1] * t_f * t_f + x_f[1] * x_f[1] * t_f * t_f) - 2 * (
                         x_0[0] - x_f[0]) - x_0[1] * t_f - x_f[1] * t_f) / (t_f * t_f)

        return u_lim

    def energy_optimal_solution(self, x_0, x_f, t_f, u_lim=None):
        if u_lim is not None:
            if u_lim > self.u_sys_lim:
                raise Exception("the provided control limit is above the system limit")
            u_max = u_lim
            u_min = -u_lim
        else:
            u_max = self.u_sys_lim
            u_min = -self.u_sys_lim

        s0 = self.switch_criteria(x_0, x_f, u_max)

        if s0 == 0:
            s0 = np.sign(x_0[1])
            u_min = u_min * 0.9
            u_max = u_max * 0.9

        if s0 > 0:
            a = u_min
            b = -x_f[1] - t_f * u_min + x_0[1]
            c = 0.5 / u_min * (x_f[1] + t_f * u_min - x_0[1]) ** 2 - t_f * (
                x_f[1] + t_f * u_min) + 0.5 * u_min * t_f ** 2 - x_0[0] + x_f[0]

            t_s = np.zeros(2)
            x_s = np.zeros([2, 2])

            if b ** 2 - 4 * a * c > 0:
                t_s[1] = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                t_s[0] = (x_f[1] + t_f * u_min - x_0[1] - t_s[1] * u_min) / u_min
                x_s[0] = np.array([0.5 * t_s[0] * t_s[0] * u_min + x_0[1] * t_s[0] + x_0[0], u_min * t_s[0] + x_0[1]])
                x_s[1] = np.array([x_s[0, 0] + (t_s[1] - t_s[0]) * x_s[0, 1], x_s[0, 1]])

            u_opt = [u_min, 0, u_max]
        elif s0 < 0:
            a = u_max
            b = -x_f[1] - t_f * u_max + x_0[1]
            c = 0.5 / u_max * (x_f[1] + t_f * u_max - x_0[1]) ** 2 - t_f * (
                x_f[1] + t_f * u_max) + 0.5 * u_max * t_f ** 2 - x_0[0] + x_f[0]

            t_s = np.zeros(2)
            x_s = np.zeros([2, 2])

            if b ** 2 - 4 * a * c > 0:
                t_s[1] = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                t_s[0] = (x_f[1] + t_f * u_max - x_0[1] - t_s[1] * u_max) / u_max
                x_s[0] = np.array([0.5 * t_s[0] * t_s[0] * u_max + x_0[1] * t_s[0] + x_0[0], u_max * t_s[0] + x_0[1]])
                x_s[1] = np.array([x_s[0, 0] + (t_s[1] - t_s[0]) * x_s[0, 1],  x_s[0, 1]])

            u_opt = [u_max, 0, u_min]
        else:
            raise Exception('not implemented yet')

        return s0, t_f, t_s, x_s, u_opt, u_max

    def energy_opt_control(self, x_0, x_f, t_f, u_lim=None):
            if x_0[0] == x_f[0] and x_0[1] == x_f[1]:  # trivial case: start == goal
                return True, Trajectory(t_f=0, t_s=0, x_s=x_0, X=np.array([x_0]), U=np.array([0]), S=np.array([0]),
                                        T=np.array([0]))

            t_f_opt = self.time_optimal_solution(x_0, x_f , u_lim)[1]
            if t_f < t_f_opt:
                return False, []  # no solution: final time greater than minimum time

            s0, t_f, t_s, x_s, u_opt, u_max = self.energy_optimal_solution(x_0, x_f, t_f, u_lim)

            if np.isnan(t_s[0]) or np.isnan(t_s[1]) or t_s[0] > t_f or t_s[0] < 0 or t_s[1] < 0 or \
                t_s[0] > t_s[1] or t_s[1] > t_f:  # no solution: switching times are unfeasible
                return False, []

            # compute parameter of switching function: s(t) = -c1 t + c2
            if s0 > 0:
                c1 = -2.0 / (t_s[0] - t_s[1])
                c2 = 1 + c1 * t_s[0]
            else:
                c1 = 2.0 / (t_s[0] - t_s[1])
                c2 = -1 + c1 * t_s[0]

            t = 0
            s0 = c2
            x = np.copy(x_0)
            u = u_opt[0]

            X = np.array([x])
            U = np.array([u])
            S = np.array([s0])
            T = np.array([t])

            while t <= t_f:
                if t < t_s[0]:
                    u = u_opt[0]
                    x_2 = u * t + x_0[1]
                    x_1 = 0.5 * t * t * u + x_0[1] * t + x_0[0]
                elif t < t_s[1]:
                    u = u_opt[1]
                    t_d = t - t_s[0]
                    x_2 = u * t_d + x_s[0][1]
                    x_1 = 0.5 * t_d * t_d * u + x_s[0][1] * t_d + x_s[0][0]
                else:
                    u = u_opt[-1]
                    t_d = t - t_s[1]
                    x_2 = u * t_d + x_s[1][1]
                    x_1 = 0.5 * t_d * t_d * u + x_s[1][1] * t_d + x_s[1][0]

                s = -c1 * t + c2

                X = np.append(X, np.array([[x_1, x_2]]), axis=0)
                U = np.append(U, u)
                S = np.append(S, s)
                T = np.append(T, t)

                t += self.dt

            return True, Trajectory(t_f=t_f, t_s=t_s, x_s=x_s, X=X, U=U, S=S, T=T)

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


class Trajectory:
    def __init__(self, t_f, t_s, x_s, X, U, S, T):
        self.t_f = t_f
        self.t_s = t_s
        self.x_s = x_s
        self.X = X
        self.U = U
        self.S = S
        self.T = T
