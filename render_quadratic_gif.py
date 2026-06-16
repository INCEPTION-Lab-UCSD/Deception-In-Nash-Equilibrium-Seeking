import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import quadratic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the quadratic game simulation as an animated GIF."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("quadratic_game_simulation.gif"),
        help="Path to the output GIF file.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=400,
        help="Frame spacing used when sampling the simulation timeline.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Playback rate for the exported GIF.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    simulation = quadratic.simulate_quadratic_game(
        **quadratic.example_6_parameters()
    )
    result = quadratic.animate_quadratic(simulation, frame_step=args.frame_step)
    writer = animation.PillowWriter(fps=args.fps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.animation.save(
        str(args.output),
        writer=writer,
        dpi=140,
    )
    plt.close(result.figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
