import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
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
    ax = fig.add_subplot(projection="3d")

    data = [Gen_RandLine(25, 3) for index in range(50)]

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

    stacked_data = np.hstack(data)
    mins = stacked_data.min(axis=1)
    maxs = stacked_data.max(axis=1)

    ax.set_xlim3d([mins[0], maxs[0]])
    ax.set_xlabel("X")

    ax.set_ylim3d([mins[1], maxs[1]])
    ax.set_ylabel("Y")

    ax.set_zlim3d([mins[2], maxs[2]])
    ax.set_zlabel("Z")

    ax.set_title("3D Test")

    line_ani = FuncAnimation(
        fig, update_lines, 100, fargs=(data, lines), interval=100, blit=False
    )

    plt.show()

    return line_ani


def gravity():
    fig, ax = plt.subplots()
    t = np.linspace(0, 3, 40)
    g = -9.81
    v0 = 12

    z = g * t**2 / 2 + v0 * t

    v02 = 5
    z2 = g * t**2 / 2 + v02 * t

    scat = ax.scatter(t[0], z[0], c="g", s=5, label=f"v0 = {v0} m/s")

    line2 = ax.plot(t[0], z2[0], label=f"v0 = {v02} m/s")[0]
    ax.set(xlim=[0, 3], ylim=[-5, 10], xlabel="Time [s]", ylabel="Height [m]")

    ax.legend()

    def update(frames):
        x = t[:frames]
        y = t[:frames]
        data = np.stack([x, y]).T
        print(data.shape)
        scat.set_offsets(data)
        line2.set_xdata(t[:frames])
        line2.set_ydata(z2[:frames])
        return (scat, line2)

    ani = animation.FuncAnimation(fig, update, frames=40, interval=30)
    plt.show()


def colored_line(x, y, c, ax, **lc_kwargs):
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    x = np.asarray(x)
    y = np.asarray(y)

    # computes midpoint for each value in the array possible
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    # lc = LineCollection(segments)
    # lc.set_array(c)
    # return ax.add_collection(lc)


if __name__ == "__main__":
    threeDPlot()
