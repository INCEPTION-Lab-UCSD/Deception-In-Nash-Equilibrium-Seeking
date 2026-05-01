import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def sin_wave():
    fig = plt.figure()

    axis = plt.axes(xlim=(0, 4), ylim=(-2, 2))

    (line,) = axis.plot([], [], lw=4)

    def animate(i):
        x = np.linspace(0, 4, 1000)

        y = np.sin(2 * np.sin(x - 0.01 * i))
        line.set_data(x, y)

        return (line,)

    anim = FuncAnimation(fig, animate, frames=200, interval=20, blit=True)

    plt.show()

    return anim


def coil():
    fig = plt.figure()
    axis = plt.axes(xlim=(-50, 50), ylim=(-50, 50))

    (line,) = axis.plot([], [], lw=2)

    xdata, ydata = [], []

    def animate(i):
        # t varies with frame number
        t = 0.1 * i

        # x, y values to be plotted
        x = t * np.sin(t)
        y = t * np.cos(t)

        xdata.append(x)
        ydata.append(y)
        line.set_data(xdata, ydata)
        return (line,)

    anim = FuncAnimation(fig, animate, frames=500, interval=20, blit=True)

    plt.show()

    return anim


def Gen_RandLine(length, dims=2):
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for i in range(1, length):
        step = (np.random.rand(dims) - 0.5) * 0.1
        lineData[:, i] = lineData[:, i - 1] + step

    return lineData


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines


def threeDPlot():
    fig = plt.figure()
    ax = Axes3D(fig)

    data = [Gen_RandLine(25, 3) for index in range(50)]

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

    ax.set_xlim3d([0.0, 1.0])
    ax.set_xlabel("X")

    ax.set_ylim3d([0.0, 1.0])
    ax.set_ylabel("Y")

    ax.set_zlim3d([0.0, 1.0])
    ax.set_zlabel("Z")

    ax.set_title("3D Test")

    line_ani = FuncAnimation(
        fig, update_lines, 25, fargs=(data, lines), interval=50, blit=False
    )

    plt.show()
