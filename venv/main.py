import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy.special as sp
import time

# For a time count on how long this code takes to run
tic = time.perf_counter()

# q1.1
def deriv(f, a, h, methods):
    """
    This function defines the forward difference method.
    :param f: The function with variable x
    :param a: number where we are computing the function at
    :param h: Step size which is usually << 1
    :param methods: inputs are forward, backward and central (only takes string input)
    :return:
    """
    # This code allows for an input of the method to use so i could write less lines of code because i am lazy
    # Using if statements to consider the method and output the appropriate finite difference method
    if str.lower(methods) == "forward":
        return (f(a + h) - f(a)) / h
    elif str.lower(methods) == "backward":
        return (f(a) - f(a - h)) / h
    elif str.lower(methods) == "central":
        return (f(a + h) - f(a - h)) / (2 * h)
    else:
        return "invalid input for method!!!"


#  1.2: each interval, for each step size and each method
#  since we know that the derivative of sin(x) = cos(x), np.cos(x) was used
x_val = np.linspace(0, 2 * m.pi, 100)

cosval = np.cos(x_val)
stepsize = [0.1/(10**i) for i in range(0, 15)]
total_vals = []
#  Triple loop to create a list of 100 points for each step size and each step size was conducted for each method
for method in ["forward", "backward", "central"]:
    Errors = []
    for i in stepsize:
        appendlist = []
        for v in x_val:
            appendlist.append(deriv(lambda x: np.sin(x), v, i, method))
        error = [abs(np.subtract(appendlist, cosval)) for a, b in zip(appendlist, cosval)]
        average = np.average(error)
        Errors.append(average)
    # print("the average errors  for each step size in the " + method + " method. \n", Errors)
    print("The errors for the ", method, " method is: ", Errors)
    total_vals += Errors
#  list manipulation to separate errors into the various methods
forwards = total_vals[0:15]
backwards = total_vals[15:30]
central = total_vals[30:]
# print(forwards, "\n",  backwards, "\n", central)


#  1.3: plot in loglog scale

fig, ax1 = plt.subplots()
lines = ax1.plot(stepsize, forwards, 'og', stepsize, backwards, 'or', stepsize, central, 'ob')
ax1.set_xlabel('x')
plt.yscale("log")
plt.xscale("log")
plt.ylabel("errors")
ax1.legend(lines, ("forwards", "backwards", "central"))
plt.title("Average errors for different derivative methods")
plt.savefig("Average Error for Derivative methods")
# plt.show()

# fig2, ax2 = plt.subplots()
# lines = ax2.plot(stepsize, backwards, 'or')
# ax2.set_xlabel('x')
# plt.yscale("log")
# plt.xscale("log")
# plt.ylabel("errors")
# ax2.legend(lines, "backwards")
# plt.title("average errors for backwards difference method")
# plt.savefig("Backwards derivative method average error")
# # plt.show()
#
# fig3, ax3 = plt.subplots()
# lines = ax3.plot(stepsize, central, 'ob')
# ax3.set_xlabel('x')
# plt.yscale("log")
# plt.xscale("log")
# plt.ylabel("errors")
# ax3.legend(lines, "central")
# plt.title("average errors for central difference method")
# plt.savefig("Central derivative method average error")
# # plt.show()


# 1.4 & 1.5 are explainy questions
# the slopes of the lines in the forwards and backwards have a decreasing error up to 10**-8 while central decreases
# until 10**-6ish. so forwards and backwards are more accurate to a lower step size

# the error starts to increase as the step size decreases because the computer is not able to be accurate to that many
# decimal places (float overflow)

#  Q2.1: defining the forward Euler method
def euler_forwards(f, y_init, step_size, interval):
    """
    euler(tau = 5, time = 25 step = 0.5, init = 100
    This function outputs the initial value problem using the Euler method
    :param f: The differential equation to solve
    :param y_init: initial value of y
    :param step_size: size of each step
    :param interval: total range to evaluate over
    :return: an array containing points along the function that we can reconstruct
    """
    # determine the number of iterations based on the total evaluation time and step size
    times = np.linspace(0, interval, int(interval / step_size) + 1)
    # create an array to store the data
    y = np.zeros(len(times))
    y[0] = y_init
    for n in range(0, len(times) - 1):
        y[n + 1] = y[n] + f(y[n], times[n]) * (times[n + 1] - times[n])
    return y


#  actual value was found to be 100*e^-t/tau which we use for comparison of the forwards and leapfrog euler methods
def act_func(t, tau):
    return 100 * m.exp(-t/tau)


if __name__ == "__main__":
    # list of steps required to reach total evaluation time
    times = np.linspace(0, 25, int(25 / 0.5) + 1)
    forward_euler = euler_forwards(lambda y, t: -(1/5)*y, 100, 0.5, 25)
    # print(forward_euler[-1], act_func(25))

    # plotting of the forward euler compared to actual result
    fig6, ax6 = plt.subplots()
    lines = ax6.plot(times, forward_euler, 'ob', 25, act_func(25, 5), '-or')
    ax6.set_xlabel('x')
    plt.ylabel("number of excited atoms")
    ax6.legend(lines, "forward euler method")
    plt.title("Forward Euler method for differentiation given function")
    plt.savefig("Forward Euler method for decay of excited atoms")
    # plt.show()

# 2.2

def euler_frog(f, y_init, step_size, interval):
    """
    essentially the same method as the forwards euler except the equation is changed
    :param f: the function to evaluate
    :param y_init: The initial value of the system at t=0
    :param step_size: The time difference between each point the system is evaluated at
    :param interval: The total time where the system is evaluated over
    :return: a list of the steps taken where the final element of the list should be the final value at the final
    interval
    """
    t = np.linspace(0, interval, int(interval / step_size) + 1)
    y = np.zeros(len(t))
    y[0] = y_init
    # we need the first 2 points and thus the forward euler method is used to determine the 2nd element in the list
    # before the leapfrog method can be used
    y[1] = y[0] + f(y[0], t[0]) * (t[1] - t[0])
    for n in range(1, len(t) - 1):
        y[n + 1] = y[n] + f(y[n], t[n]) * (t[n + 1] - t[n])
    return y


if __name__ == "__main__":
    # need to create an array for the step sizes we are using to evaluate the methods
    times = np.linspace(0, 25, int(25 / 0.5) + 1)
    LF_Euler = euler_frog(lambda y, t: -(1/5)*y, 100, 0.5, 25)

    # plotting the values
    fig6, ax6 = plt.subplots()
    lines = ax6.plot(times, LF_Euler, 'ob', 25, act_func(25, 5), '-or')
    ax6.set_xlabel('x')
    plt.ylabel("number of excited atoms")
    ax6.legend(lines, "forward euler method")
    plt.title("Leap Frog Euler method for decay of excited atoms")
    plt.savefig("Leap Frog Euler method for decay of excited atoms")
    # plt.show()


#  2.3, plot both forwards and leap frog method vs actual value.


if __name__ == "__main__":
    step = [1/2**i for i in range(0, 21)]
    forwards_error = []
    frog_error = []
    for i in range(len(step)):
        forwards_error.append(abs((euler_forwards(lambda y, t: -(1/5)*y, 100, step[i], 25)[-1] - act_func(25, 5))/act_func(25, 5)))
        frog_error.append(abs((euler_frog(lambda y, t: -(1/5)*y, 100, step[i], 25)[-1] - act_func(25, 5))/act_func(25, 5)))
    # print(forwards_error, "\n", frog_error)
        print("Are the forward and leapfrog methods the same: ", euler_forwards(lambda y, t: -(1/5)*y, 100, step[i], 25) == euler_frog(lambda y, t: -(1/5)*y, 100, step[i], 25))

# Plot for the leapfrog and forwards methods
    fig4, ax4 = plt.subplots()
    lines = ax4.plot(step, forwards_error, 'og', step, frog_error, 'oy')
    ax4.set_xlabel('t')
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("errors")
    ax4.legend(lines, ("forwards euler error", "Leap frog error"))
    plt.title("forwards error and the leap frog method error ")
    plt.savefig("Forwards Euler and Leap Frog Euler method errors")
    # plt.show()

# 2.4 how do the two methods compare for different step sizes
# The two methods are seen to be exactly the same visually but there are minute differences in the fractional
# errors with the leapfrog method being slightly better

#  3.1
def rk_order2(f, x0, y0, x, h):
    """
    The runge-kutta method solves for a specified point along an arbitrary function
    :param f: The differential equation to undergo the runge kutta method. must be written with variables x and y
    :param x0: initial value of x
    :param y0: initial value of y at x0
    :param x: the value or values of x to find the y for
    :param h: step size
    :return: The y value at the specified x we wish to evaluate a function for
    """
    n = int((x - x0) / h) + 1  # number of iterations to reach required point
    y = y0
    for i in range(1, n):
        # define the runge kutta variables used in finding the next value
        k1 = f(x0, y)
        k2 = f(x0 + 0.5 * h, y + 0.5 * k1 * h)
        # using the runge kutta variables to find the next point to use in the iteration process
        y = y + h * k2
        x0 = x0 + h
    return y


def errfunc(t):
    return ((m.sqrt(m.pi)) / 2) * m.exp(t ** 2) * sp.erf(t)


# if __name__ == "__main__":
    # test = rk_order2(lambda x, y: 1+ 2*x*y, 0, 0, 1, 0.01)
#     print(test)


#  3.2
if __name__ == "__main__":
    step = [1 / 2 ** i for i in range(0, 21)]
    rkerr = []
    for i in range(len(step)):
        rkerr.append(abs((rk_order2(lambda x, y: 1 + 2 * x * y, 0, 0, 25, step[i]) - errfunc(25)) / errfunc(25)))
    # print(rkerr)

    fig5, ax5 = plt.subplots()
    lines = ax5.plot(step, rkerr, 'ob')
    ax5.set_xlabel('t')
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("errors")
    ax5.legend(lines, "runge-kutta error")
    plt.title("Runge-Kutta error ")
    plt.savefig("Runge-Kutta Error")
    # plt.show()

toc = time.perf_counter()
print(toc-tic)

#  4.1 (for fund)
# from scipy.integrate import solve_ivp
#
#
# def f(t, y):
#     return [1 + 2 * t * y[0]]
#
#
# tstart, tend = 0.0, 1.0
# y0 = [0.0]
# eps = 1e-6
# tvals = np.linspace(tstart, tend, 100)
#
# res = solve_ivp(f, (tstart, tend), y0, t_eval=tvals, rtol=eps)
# yvals = res.y[0]
#
# fig, ax = plt.subplots()
# ax.plot(tvals, yvals, '-x')
# ax.set_xlabel('$t$'), ax.set_ylabel('$y(t)$')
