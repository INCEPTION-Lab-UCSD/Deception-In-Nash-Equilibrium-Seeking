# Deception in Nash Equilibrium Seeking

This repository contains Python simulations and visualization scripts for
deception in Nash equilibrium seeking games. The source code is kept in
`src/`, and pre-rendered figures and animations are kept in `animations/`.

## Repository layout

```text
.
├── animations/   # Pre-rendered GIF and PNG outputs
├── src/          # Simulation modules and rendering scripts
├── README.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Setup

The project uses Python 3.12. Create a virtual environment and install the
Python dependencies with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run simulations

Run commands from the repository root.

```bash
python src/main.py
```

`src/main.py` is the interactive entry point. It currently runs the
second-order mutual deception duopoly example and opens the plot window with
Matplotlib. Edit the `main()` function in `src/main.py` to switch between the
duopoly, mutual deception, and quadratic examples.

The render scripts export animations directly to files:

```bash
python src/render_aggregative_gif.py --output animations/aggregative_game_simulation.gif
python src/render_aggregative_convergence_gif.py --output animations/aggregative_game_convergence.gif
python src/render_quadratic_gif.py --output animations/quadratic_game_simulation.gif
```

Each render script supports `--help`, `--output`, `--frame-step`, and `--fps`.

## View plots and animations

The repository includes pre-rendered outputs:

- `animations/aggregative_game_simulation.gif`
- `animations/aggregative_game_convergence.gif`
- `animations/quadratic_game_simulation.gif`
- `animations/duopoly_game_payoff_curves.gif`
- `animations/Duopoly_with_Deception_Simulation_Reaction_Curves.gif`
- `animations/dupoly_with_deception_simulation.gif`
- `animations/first_order_mutual_deception.gif`
- `animations/second_order_mutual_deception.gif`
- `animations/duopoly_plots.png`

Open these files directly from the `animations/` directory, or re-run the
render commands above to regenerate selected outputs.

## Citation
```bibtex
@misc{tang2025deceptionnashequilibriumseeking,
      title={Deception in Nash Equilibrium Seeking}, 
      author={Michael Tang and Umar Javed and Xudong Chen and Miroslav Krstic and Jorge I. Poveda},
      year={2025},
      eprint={2407.05168},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2407.05168}, 
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
