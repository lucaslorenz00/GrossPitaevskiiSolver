"""
utils/io.py

Generated with AI, does not interact with the phyiscal simulation,
only for saving and loading data.

Save and load simulation checkpoints

Saves GPState objects to NumPy .npz files (compressed archives).
Each checkpoint stores:
  - The wavefunction arrays (psi_r, psi_i)
  - Grid parameters (N, L)
  - Metadata (n_comp, time, label)
  - Optional: the full history list as a JSON string

Usage
-----
    from gp_solver.utils.io import save_checkpoint, load_checkpoint

    # Save after a run
    save_checkpoint(prop.state, "soliton_t50.npz")

    # Load and continue
    state = load_checkpoint("soliton_t50.npz", grid)

    # Save history as csv file
    save_history_csv(state, "history.csv")
"""


from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from gp_solver.core.grid  import Grid
from gp_solver.core.state import GPState


def save_checkpoint(
    state:    GPState,
    filepath: str,
    history:  bool = True,
) -> None:
    """
    Save a GPState to a compressed .npz file.

    Parameters
    ----------
    state    : GPState  — state to save
    filepath : str      — output path (will add .npz if missing)
    history  : bool     — whether to save state.history (can be large)
    """
    path = Path(filepath)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    save_dict = {
        "psi_r":  state.psi_r,
        "psi_i":  state.psi_i,
        "N":      np.array([state.grid.N]),
        "L":      np.array([state.grid.L]),
        "n_comp": np.array([state.n_comp]),
        "t":      np.array([state.t]),
        "label":  np.array([state.label]),
    }

    if history and state.history:
        try:
            history_json = json.dumps(state.history)
            save_dict["history_json"] = np.array([history_json])
        except (TypeError, ValueError):
            pass   # skip history if not serializable

    np.savez_compressed(str(path), **save_dict)
    print(f"[io] Saved checkpoint -> {path}  "
          f"(n_comp={state.n_comp}, t={state.t:.4f})")


def load_checkpoint(
    filepath: str,
    grid:     Grid = None,
) -> GPState:
    """
    Load a GPState from a .npz checkpoint file.

    Parameters
    ----------
    filepath : str        — path to .npz file
    grid     : Grid       — if None, grid is reconstructed from saved N, L

    Returns
    -------
    GPState with wavefunction, time, and history restored.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data   = np.load(str(path), allow_pickle=False)
    N      = int(data["N"][0])
    L      = float(data["L"][0])
    n_comp = int(data["n_comp"][0])
    t      = float(data["t"][0])
    label  = str(data["label"][0])

    if grid is None:
        grid = Grid(N=N, L=L)
    elif grid.N != N or abs(grid.L - L) > 1e-10:
        raise ValueError(
            f"Provided grid (N={grid.N}, L={grid.L}) does not match "
            f"checkpoint (N={N}, L={L}). Pass grid=None to auto-reconstruct."
        )

    state = GPState(
        grid   = grid,
        n_comp = n_comp,
        psi_r  = data["psi_r"].copy(),
        psi_i  = data["psi_i"].copy(),
        t      = t,
        label  = label,
    )

    if "history_json" in data:
        try:
            state.history = json.loads(str(data["history_json"][0]))
        except (json.JSONDecodeError, KeyError):
            state.history = []

    print(f"[io] Loaded checkpoint ← {path}  "
          f"(n_comp={n_comp}, t={t:.4f})")
    return state


def save_history_csv(state: GPState, filepath: str) -> None:
    """
    Export the scalar entries of state.history to a CSV file.

    Useful for plotting convergence, norm conservation, etc. in
    external tools (Excel, gnuplot, …).

    Parameters
    ----------
    state    : GPState
    filepath : str — output .csv path
    """
    if not state.history:
        print("[io] History is empty — nothing to export.")
        return

    path = Path(filepath)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")

    # Collect all scalar keys across all snapshots
    all_keys = set()
    for snap in state.history:
        for k, v in snap.items():
            if isinstance(v, (int, float)):
                all_keys.add(k)
    all_keys = sorted(all_keys)

    with open(str(path), "w") as fh:
        fh.write(",".join(all_keys) + "\n")
        for snap in state.history:
            row = []
            for k in all_keys:
                val = snap.get(k, "")
                row.append(str(val) if val != "" else "")
            fh.write(",".join(row) + "\n")

    print(f"[io] History exported -> {path}  ({len(state.history)} rows)")
