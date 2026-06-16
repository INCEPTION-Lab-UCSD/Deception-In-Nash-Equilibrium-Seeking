import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import aggregative


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the aggregative game reaction-curve simulation as a GIF."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aggregative_game_simulation.gif"),
        help="Path to the output GIF file.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=40,
        help="Frame spacing used when sampling the simulation timeline.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Playback rate for the exported GIF.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    simulation = aggregative.simulation_aggregative(
        x0=np.array([-0.2, 0.1], dtype=float),
        a=0.015,
        k=0.01,
        omega_1=470.75,
        omega_2=330.0,
        J_2_ref=-0.1,
        epsilon=0.001,
        horizon=150.0,
        dt=0.05,
    )
    simulation.delta = np.linspace(-3.0, 3.0, len(simulation.time), dtype=float)

    anim, figure, _axes = aggregative.animate_reaction_curves(
        simulation=simulation,
        frame_step=args.frame_step,
        interval=40,
        repeat_delay=1200,
        delta_limits=(-3.0, 3.0),
    )

    writer = animation.PillowWriter(fps=args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.output), writer=writer, dpi=140)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
