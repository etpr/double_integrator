import matplotlib.pyplot as plt
import numpy as np
from double_integrator import DoubleIntegrator

di = DoubleIntegrator(u_sys_lim=1.0, dt=0.001)

x_10 = 0.5
x_20 = 1.0
x_1f = -0.5
x_2f = 1.0

t_f, t_s, x_1s, x_2s, X, U, S, T = di.time_opt_control(x_10=x_10, x_20=x_20, x_1f=x_1f, x_2f=x_2f)

plt.figure(1)
ax = plt.subplot(411)
ax.grid(True, which='both')
plt.xlabel('t')
plt.ylabel('x_1')
plt.plot(T, X[:, 0])
ax = plt.subplot(412)
ax.grid(True, which='both')
plt.xlabel('t')
plt.ylabel('x_2')
plt.plot(T, X[:, 1])
ax = plt.subplot(413)
ax.grid(True, which='both')
plt.plot(T, U)
plt.xlabel('t')
plt.ylabel('u')
ax = plt.subplot(414)
ax.grid(True, which='both')
plt.plot(T, S)
plt.xlabel('t')
plt.ylabel('s')

plt.figure(2)
plot_idx = np.bitwise_and(np.bitwise_not(np.isclose(S, 0)), S > 0)
plt.plot(X[plot_idx, 0], X[plot_idx, 1], 'r')
plot_idx = np.bitwise_and(np.bitwise_not(np.isclose(S, 0)), S < 0)
plt.plot(X[plot_idx, 0], X[plot_idx, 1], 'b')
plt.plot(X[np.isclose(S, 0), 0], X[np.isclose(S, 0), 1], 'k')

plt.plot(x_10, x_20, 'co')
plt.plot(x_1f, x_2f, 'go')
plt.plot(x_1s, x_2s, 'mo')
plt.legend(["s > 0", "s < 0", "s = 0", 'x_0', 'x_f', 'x_s'])
plt.ylim([-2, 2])
plt.xlim([-2, 2])
ax = plt.gca()
ax.set_aspect('equal')
ax.grid(True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.xlabel('x_1')
plt.ylabel('x_2')

plt.show()
