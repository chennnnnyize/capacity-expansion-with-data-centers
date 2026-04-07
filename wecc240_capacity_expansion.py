"""CVXPY capacity expansion with delayable and geographically shiftable data-center load
for the WECC-240 case from eudoxys/wecc240.

Notes
-----
1) This uses the MATPOWER-style case in `wecc240_2011.py`.
2) Hourly load is read from `wecc_load.csv` with columns like `Load_AESO_2009.dat`, ...
3) By default, the 40-column WECC hourly load file is aggregated to a single system load
   profile and then distributed to buses proportionally to the case's base PD. This is the
   safest option if there is no authoritative mapping from load-area columns to WECC-240 buses.
4) You can replace `build_bus_load_matrix_from_total_profile` with a bus-area mapping later.
5) Full-year CVXPY on WECC-240 can be very large. Start with a shorter horizon or rolling windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types
import importlib.util
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt


# -----------------------------
# MATPOWER column indices
# -----------------------------
BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
BUS_AREA = 6
VM = 7
VA = 8
BASE_KV = 9
ZONE = 10
VMAX = 11
VMIN = 12

F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
RATE_A = 5
TAP = 8
SHIFT = 9
BR_STATUS = 10
ANGMIN = 11
ANGMAX = 12

GEN_BUS = 0
PG = 1
QG = 2
QMAX = 3
QMIN = 4
VG = 5
MBASE = 6
GEN_STATUS = 7
PMAX = 8
PMIN = 9

MODEL = 0
STARTUP = 1
SHUTDOWN = 2
NCOST = 3
COST = 4

# DC line columns (MATPOWER)
DC_F_BUS = 0
DC_T_BUS = 1
DC_STATUS = 2
DC_PF = 3
DC_PT = 4
DC_QF = 5
DC_QT = 6
DC_VF = 7
DC_VT = 8
DC_PMIN = 9
DC_PMAX = 10


@dataclass
class FlexConfig:
    dc_delay_buses: list[int]
    geo_send_buses: list[int]
    geo_receive_buses: list[int]
    delay_share_of_system_load: float = 0.03
    geo_share_of_system_load: float = 0.04
    delay_window_hours: int = 6
    max_send_fraction: float = 0.50
    max_receive_fraction: float = 0.40
    shift_budget_fraction: float = 0.80


@dataclass
class SolveConfig:
    load_csv: str = "wecc_load.csv"
    wecc_case_py: str = "wecc240_2011.py"
    horizon_hours: int | None = None   # None -> full file
    start_hour: int = 0
    delay_penalty: float = 1.0
    shift_penalty: float = 1.0
    slack_penalty: float | None = 1e5
    expansion_cost_scale: float = 200.0
    expansion_limit_fraction: float = 0.30


# -----------------------------
# Case loading
# -----------------------------
def load_wecc_case(case_file: str | Path) -> dict:
    """Load the eudoxys WECC-240 python case file without requiring pypower_sim."""
    case_file = str(case_file)
    stub = types.ModuleType("pypower_sim")
    stub.PPModel = object
    stub.PPSolver = object
    sys.modules["pypower_sim"] = stub

    spec = importlib.util.spec_from_file_location("wecc240_case_module", case_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load case file: {case_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "wecc240_2011"):
        return module.wecc240_2011()
    if hasattr(module, "WECC240_2011"):
        return module.WECC240_2011()
    raise AttributeError("Could not find `wecc240_2011` or `WECC240_2011` in the case file.")


# -----------------------------
# Load profile processing
# -----------------------------
def read_wecc_hourly_load(csv_file: str | Path, start_hour: int = 0, horizon_hours: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    load_cols = [c for c in df.columns if c.startswith("Load_")]
    if not load_cols:
        raise ValueError("No columns starting with 'Load_' were found in wecc_load.csv")

    if "Index" in df.columns:
        df = df.sort_values("Index")
    df = df.reset_index(drop=True)

    end_hour = None if horizon_hours is None else start_hour + horizon_hours
    out = df.iloc[start_hour:end_hour].copy()
    if out.empty:
        raise ValueError("Selected hour range is empty.")
    return out[[*load_cols]]


def build_bus_load_matrix_from_total_profile(case: dict, hourly_df: pd.DataFrame,
                                             target: str = "average",
                                             target_scale: float = 1.0) -> np.ndarray:
    bus = case["bus"].astype(float)
    base_pd = np.maximum(bus[:, PD], 0.0)
    base_total = float(base_pd.sum())
    if base_total <= 0:
        raise ValueError("Base-case total PD is non-positive.")

    weights = base_pd / base_total
    raw_total = hourly_df.astype(float).sum(axis=1).to_numpy()

    if target == "average":
        scale = (base_total * target_scale) / np.mean(raw_total)
    elif target == "peak":
        scale = (base_total * target_scale) / np.max(raw_total)
    else:
        raise ValueError("target must be 'average' or 'peak'")

    total_hourly = scale * raw_total
    L_base = weights[:, None] * total_hourly[None, :]

    return L_base


# -----------------------------
# Network matrices from MATPOWER case
# -----------------------------
def build_dc_network(case: dict):
    baseMVA = float(case["baseMVA"])
    bus = case["bus"].astype(float)
    branch = case["branch"].astype(float)
    gen = case["gen"].astype(float)
    dcline = case.get("dcline", np.zeros((0, 17))).astype(float)
    gencost = case["gencost"].astype(float)

    active_branch = branch[branch[:, BR_STATUS] > 0.0]
    active_gen = gen[gen[:, GEN_STATUS] > 0.0]
    active_dc = dcline[dcline[:, DC_STATUS] > 0.0] if dcline.size else np.zeros((0, 17))

    bus_ids = bus[:, BUS_I].astype(int)
    bus_map = {bus_id: i for i, bus_id in enumerate(bus_ids)}

    N = len(bus_ids)
    L = active_branch.shape[0]
    G = active_gen.shape[0]
    H = active_dc.shape[0]

    A_line = np.zeros((L, N))
    susceptance = np.zeros(L)
    Fmax = np.zeros(L)

    for ell, row in enumerate(active_branch):
        i = bus_map[int(row[F_BUS])]
        j = bus_map[int(row[T_BUS])]
        x = row[BR_X]
        tap = row[TAP] if row[TAP] != 0 else 1.0
        b = 1.0 / x / tap if abs(x) > 1e-9 else 0.0
        A_line[ell, i] = 1.0
        A_line[ell, j] = -1.0
        susceptance[ell] = b
        rate_a = row[RATE_A]
        Fmax[ell] = rate_a / baseMVA if rate_a > 0 else 10.0

    Gmap = np.zeros((N, G))
    Pmax = np.zeros(G)
    Pmin = np.zeros(G)
    op_cost = np.zeros(G)
    gen_bus_idx = np.zeros(G, dtype=int)

    for g, row in enumerate(active_gen):
        i = bus_map[int(row[GEN_BUS])]
        Gmap[i, g] = 1.0
        gen_bus_idx[g] = i
        Pmax[g] = row[PMAX] / baseMVA
        Pmin[g] = max(row[PMIN], 0.0) / baseMVA
        # linear term from gencost polynomial if available
        if int(gencost[g, MODEL]) == 2:
            ncost = int(gencost[g, NCOST])
            coeffs = gencost[g, COST:COST+ncost]
            if ncost >= 2:
                op_cost[g] = coeffs[-2]  # linear coefficient c1
            elif ncost == 1:
                op_cost[g] = coeffs[-1]
            else:
                op_cost[g] = 0.0
        else:
            op_cost[g] = 1.0

    # HVDC links as controllable directed transfers h_t with pmin <= h_t <= pmax
    Hmap_from = np.zeros((N, H))
    Hmap_to = np.zeros((N, H))
    Hmin = np.zeros(H)
    Hmax = np.zeros(H)
    for h, row in enumerate(active_dc):
        i = bus_map[int(row[DC_F_BUS])]
        j = bus_map[int(row[DC_T_BUS])]
        Hmap_from[i, h] = 1.0
        Hmap_to[j, h] = 1.0
        Hmin[h] = row[DC_PMIN] / baseMVA
        Hmax[h] = row[DC_PMAX] / baseMVA

    return {
        "baseMVA": baseMVA,
        "bus": bus,
        "branch": active_branch,
        "gen": active_gen,
        "dcline": active_dc,
        "bus_ids": bus_ids,
        "bus_map": bus_map,
        "A_line": A_line,
        "B_line": np.diag(susceptance),
        "K": A_line.T,
        "Fmax": Fmax,
        "Gmap": Gmap,
        "Pmax": Pmax,
        "Pmin": Pmin,
        "op_cost": op_cost,
        "gen_bus_idx": gen_bus_idx,
        "Hmap_from": Hmap_from,
        "Hmap_to": Hmap_to,
        "Hmin": Hmin,
        "Hmax": Hmax,
        "N": N,
        "L": L,
        "G": G,
        "H": H,
    }


# -----------------------------
# Flexible DC load construction
# -----------------------------
def build_flexible_data_center_loads(L_base: np.ndarray, bus_ids: np.ndarray, flex: FlexConfig):
    N, T = L_base.shape
    bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}

    delay_idx = [bus_id_to_idx[b] for b in flex.dc_delay_buses]
    send_idx = [bus_id_to_idx[b] for b in flex.geo_send_buses]
    recv_idx = [bus_id_to_idx[b] for b in flex.geo_receive_buses]

    if len(delay_idx) == 0:
        raise ValueError("No valid delay buses found in case for dc_delay_buses.")
    if len(send_idx) == 0 or len(recv_idx) == 0:
        raise ValueError("Need at least one valid send bus and one valid receive bus for geo shifting.")

    system_total = L_base.sum(axis=0)

    # Delayable arrivals U: split a fraction of system load across selected buses
    U = np.zeros((len(delay_idx), T))
    delay_total = flex.delay_share_of_system_load * system_total
    for k, bi in enumerate(delay_idx):
        w = L_base[bi, :] / np.maximum(L_base[delay_idx, :].sum(axis=0), 1e-8)
        U[k, :] = delay_total * w

    # Mapping Afix from delay classes to buses
    D = len(delay_idx)
    Afix = np.zeros((N, D))
    for k, bi in enumerate(delay_idx):
        Afix[bi, k] = 1.0

    # Baseline geographically shiftable load located initially at send buses
    L_geo = np.zeros((N, T))
    geo_total = flex.geo_share_of_system_load * system_total
    send_base = L_base[send_idx, :]
    send_weights = send_base / np.maximum(send_base.sum(axis=0), 1e-8)
    L_geo[send_idx, :] = send_weights * geo_total

    delta_lower = np.zeros((N, T))
    delta_upper = np.zeros((N, T))

    # send buses can only reduce geo load
    delta_lower[send_idx, :] = -flex.max_send_fraction * L_geo[send_idx, :]
    delta_upper[send_idx, :] = 0.0

    # receive buses can only absorb shifted load
    recv_cap = flex.max_receive_fraction * geo_total
    for bi in recv_idx:
        delta_lower[bi, :] = 0.0
        delta_upper[bi, :] = recv_cap / max(len(recv_idx), 1)

    Gamma = flex.shift_budget_fraction * np.sum(np.maximum(delta_upper, 0.0), axis=0)

    return {
        "U": U,
        "Afix": Afix,
        "L_geo": L_geo,
        "delta_lower": delta_lower,
        "delta_upper": delta_upper,
        "Gamma": Gamma,
        "delay_bus_idx": np.array(delay_idx, dtype=int),
        "send_bus_idx": np.array(send_idx, dtype=int),
        "recv_bus_idx": np.array(recv_idx, dtype=int),
        "W": np.full(D, flex.delay_window_hours, dtype=int),
    }


# -----------------------------
# Optimization model
# -----------------------------
def build_wecc_cvxpy_model(case_data: dict,
                           L_base: np.ndarray,
                           flex_data: dict,
                           cfg: SolveConfig):
    N, T = L_base.shape
    print("Base Load", L_base)
    #plt.plot(L_base[:,0])
    #plt.show()
    G = case_data["G"]
    L = case_data["L"]
    H = case_data["H"]
    D = flex_data["U"].shape[0]

    A_line = case_data["A_line"]
    B_line = case_data["B_line"]
    K = case_data["K"]
    Gmap = case_data["Gmap"]
    Pmax = case_data["Pmax"]*1.05
    Pmin = case_data["Pmin"]
    op_cost = case_data["op_cost"]
    Fmax = case_data["Fmax"]*0.85
    Hmap_from = case_data["Hmap_from"]
    Hmap_to = case_data["Hmap_to"]
    Hmin = case_data["Hmin"]
    Hmax = case_data["Hmax"]

    U = flex_data["U"]
    Afix = flex_data["Afix"]
    L_geo = flex_data["L_geo"]
    delta_lower = flex_data["delta_lower"]
    delta_upper = flex_data["delta_upper"]
    Gamma = flex_data["Gamma"]
    W = flex_data["W"]

    # Expansion allowed at every active generator bus, capped as a fraction of existing PMAX.
    Xmax = cfg.expansion_limit_fraction * Pmax
    capex = cfg.expansion_cost_scale * np.ones(G)

    x = cp.Variable(G, nonneg=True)
    p = cp.Variable((G, T))
    theta = cp.Variable((N, T))
    f = cp.Variable((L, T))
    h = cp.Variable((H, T)) if H > 0 else None

    s = cp.Variable((D, T), nonneg=True)
    b = cp.Variable((D, T), nonneg=True)

    delta = cp.Variable((N, T))
    z_shift = cp.Variable((N, T), nonneg=True)

    load_shed = cp.Variable((N, T), nonneg=True) if cfg.slack_penalty is not None else None

    cons = []
    cons += [x <= Xmax]
    cons += [p >= Pmin[:, None], p <= (Pmax + x)[:, None]]

    for t in range(T):
        cons += [
            f[:, t] == B_line @ (A_line @ theta[:, t]),
            f[:, t] <= Fmax,
            f[:, t] >= -Fmax,
            theta[0, t] == 0.0,
        ]
        if H > 0:
            cons += [h[:, t] >= Hmin, h[:, t] <= Hmax]

    for t in range(T):
        if t == 0:
            cons += [b[:, t] == U[:, t] - s[:, t], s[:, t] <= U[:, t]]
        else:
            cons += [b[:, t] == b[:, t - 1] + U[:, t] - s[:, t], s[:, t] <= b[:, t - 1] + U[:, t]]
    cons += [b[:, T - 1] == 0.0]

    for d in range(D):
        for t in range(W[d], T):
            cons += [cp.sum(s[d, :t + 1]) >= float(np.sum(U[d, :t - W[d] + 1]))]

    cons += [delta <= delta_upper, delta >= delta_lower]
    cons += [z_shift >= delta, z_shift >= -delta]

    for t in range(T):
        cons += [cp.sum(delta[:, t]) == 0.0]
        cons += [cp.sum(z_shift[:, t]) <= Gamma[t]]

    for t in range(T):
        net_inj = Gmap @ p[:, t] - L_base[:, t] - Afix @ s[:, t] - (L_geo[:, t] + delta[:, t])
        if H > 0:
            # positive h means transfer from from-bus to to-bus
            net_inj += -Hmap_from @ h[:, t] + Hmap_to @ h[:, t]
        if load_shed is not None:
            net_inj += load_shed[:, t]
        cons += [net_inj == K @ f[:, t]]

    obj = capex @ x + cp.sum(cp.multiply(op_cost[:, None], p))
    obj += cfg.delay_penalty * cp.sum(b)
    obj += cfg.shift_penalty * cp.sum(z_shift)
    if load_shed is not None:
        obj += cfg.slack_penalty * cp.sum(load_shed)

    problem = cp.Problem(cp.Minimize(obj), cons)
    vars_dict = {
        "x": x, "p": p, "theta": theta, "f": f, "h": h,
        "s": s, "b": b,
        "delta": delta, "z_shift": z_shift,
        "load_shed": load_shed,
        "L_geo": L_geo, "L_base": L_base, "U": U,
    }
    return problem, vars_dict


# -----------------------------
# Solve / plotting
# -----------------------------
def solve_problem(problem: cp.Problem, verbose: bool = True) -> str:
    installed = cp.installed_solvers()
    print("Installed CVXPY solvers:", installed)
    candidates = ["CLARABEL", "ECOS", "SCS", "SCIPY", "OSQP"]
    last_err = None
    for s in candidates:
        if s in installed:
            try:
                print(f"Trying solver: {s}")
                if s == "SCS":
                    problem.solve(solver=s, verbose=verbose, eps=1e-4, max_iters=20000)
                elif s == "OSQP":
                    problem.solve(solver=s, verbose=verbose, eps_abs=1e-5, eps_rel=1e-5)
                else:
                    problem.solve(solver=s, verbose=verbose)
                print("Status:", problem.status)
                print("Objective:", problem.value)
                return s
            except Exception as e:
                print(f"Solver {s} failed: {e}")
                last_err = e
    raise RuntimeError(f"All attempted solvers failed. Last error: {last_err}")


def sweep_delay_window(cfg: SolveConfig, flex_template: FlexConfig,
                       delay_windows=[0, 12],
                       plot_bus_idx: int = 0,
                       plot_day: int = 0):
    """
    Sweep delay window from 0 to 12 hours and compare planning / operational outcomes.

    Parameters
    ----------
    cfg : SolveConfig
    flex_template : FlexConfig
        Base flexibility configuration. Only delay_window_hours is changed.
    delay_windows : iterable
        Windows to test, default 0..12
    plot_bus_idx : int
        Which delay class / bus to visualize in subplot (c).
    plot_day : int
        Which day to visualize (0 -> first 24 hours).
    """
    results = []

    for W in delay_windows:
        print(f"\n=== Solving delay_window_hours = {W} ===")

        flex = FlexConfig(
            dc_delay_buses=flex_template.dc_delay_buses,
            geo_send_buses=flex_template.geo_send_buses,
            geo_receive_buses=flex_template.geo_receive_buses,
            delay_share_of_system_load=flex_template.delay_share_of_system_load,
            geo_share_of_system_load=flex_template.geo_share_of_system_load,
            delay_window_hours=W,
            max_send_fraction=flex_template.max_send_fraction,
            max_receive_fraction=flex_template.max_receive_fraction,
            shift_budget_fraction=flex_template.shift_budget_fraction,
        )

        problem, vars_dict, case_data, flex_data = run_wecc240_capacity_expansion(cfg, flex)

        baseMVA = case_data["baseMVA"]

        x = np.asarray(vars_dict["x"].value).ravel() * baseMVA
        total_added_gen = float(np.sum(x))
        total_cost = float(problem.value)

        # backlog integral = total delayed workload-hours
        b = np.asarray(vars_dict["b"].value) * baseMVA
        total_delayed_workload_hours = float(np.sum(b))

        # pick one delay bus / class for visualization
        U = np.asarray(vars_dict["U"]) * baseMVA
        s = np.asarray(vars_dict["s"].value) * baseMVA

        d_idx = min(plot_bus_idx, U.shape[0] - 1)
        start = 24 * plot_day
        end = min(start + 24, U.shape[1])

        row = {
            "delay_window": W,
            "total_added_gen_mw": total_added_gen,
            "total_cost": total_cost,
            "total_delayed_workload_hours": total_delayed_workload_hours,
            "case_data": case_data,
            "flex_data": flex_data,
            "U_day": U[d_idx, start:end].copy(),
            "s_day": s[d_idx, start:end].copy(),
            "delay_bus_id": int(case_data["bus_ids"][flex_data["delay_bus_idx"][d_idx]]),
        }
        results.append(row)

        print(
            f"W={W:2d} | "
            f"added_gen={total_added_gen:10.3f} MW | "
            f"total_cost={total_cost:14.4f} | "
            f"delayed_workload_hours={total_delayed_workload_hours:12.4f}"
        )

    plot_delay_window_sweep(results, plot_day=plot_day)
    return results

def plot_delay_window_sweep(results, plot_day: int = 0):
    windows = [r["delay_window"] for r in results]
    added_gen = [r["total_added_gen_mw"] for r in results]
    total_cost = [r["total_cost"] for r in results]
    delayed_wh = [r["total_delayed_workload_hours"] for r in results]

    # representative windows for workload profile
    repr_windows = [0, 6, 12]
    repr_rows = {r["delay_window"]: r for r in results if r["delay_window"] in repr_windows}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # (a) added generation
    axes[0, 0].plot(windows, added_gen, marker="o", color="red")
    axes[0, 0].set_title("(a) Total added generation")
    axes[0, 0].set_xlabel("Delay window (hours)")
    axes[0, 0].set_ylabel("MW")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    # (b) total cost
    axes[0, 1].plot(windows, total_cost, marker="o", color="red")
    axes[0, 1].set_title("(b) Total cost")
    axes[0, 1].set_xlabel("Delay window (hours)")
    axes[0, 1].set_ylabel("Cost")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    # (c) one bus original vs served workload over 24h
    ax = axes[1, 0]
    plotted_original = False
    for W in repr_windows:
        if W not in repr_rows:
            continue
        row = repr_rows[W]
        hrs = np.arange(len(row["U_day"]))
        if not plotted_original:
            ax.plot(hrs, row["U_day"], linestyle="--", linewidth=2, label="Original arrival")
            plotted_original = True
        ax.plot(hrs, row["s_day"], label=f"Served workload, W={W}")
    if repr_rows:
        any_row = next(iter(repr_rows.values()))
        ax.set_title(f"(c) Delay bus {any_row['delay_bus_id']} workload over 24 hours")
    else:
        ax.set_title("(c) One bus workload over 24 hours")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MW")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) total delayed time * workload
    axes[1, 1].bar(windows, delayed_wh, color="red")
    axes[1, 1].set_title("(d) Total delayed time × workload")
    axes[1, 1].set_xlabel("Delay window (hours)")
    axes[1, 1].set_ylabel("MW·h")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.show()


def plot_results(case_data: dict, vars_dict: dict, flex_data: dict, out_png: str | None = None):
    baseMVA = case_data["baseMVA"]

    x = np.asarray(vars_dict["x"].value).ravel() * baseMVA
    U = np.asarray(vars_dict["U"]) * baseMVA
    s = np.asarray(vars_dict["s"].value) * baseMVA
    b = np.asarray(vars_dict["b"].value) * baseMVA
    delta = np.asarray(vars_dict["delta"].value) * baseMVA
    L_geo = np.asarray(vars_dict["L_geo"]) * baseMVA

    hours = np.arange(U.shape[1])

    send_idx = np.asarray(flex_data["send_bus_idx"], dtype=int)
    recv_idx = np.asarray(flex_data["recv_bus_idx"], dtype=int)
    bus_ids = np.asarray(case_data["bus_ids"])

    moved = 0.5 * np.sum(np.abs(delta), axis=0)
    send_before = np.sum(L_geo[send_idx, :], axis=0)
    send_after = np.sum(L_geo[send_idx, :] + delta[send_idx, :], axis=0)
    recv_before = np.sum(L_geo[recv_idx, :], axis=0)
    recv_after = np.sum(L_geo[recv_idx, :] + delta[recv_idx, :], axis=0)
    delay_arrival = np.sum(U, axis=0)
    delay_served = np.sum(s, axis=0)
    backlog = np.sum(b, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # (a) Added generation capacity
    axes[0, 0].bar(np.arange(len(x)), x)
    axes[0, 0].set_title("(a) Added generation capacity")
    axes[0, 0].set_xlabel("Generator index")
    axes[0, 0].set_ylabel("MW")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    # (b) Aggregate send/receive geo load before and after
    axes[0, 1].plot(hours, send_before, label="Send buses: before")
    axes[0, 1].plot(hours, send_after, label="Send buses: after")
    axes[0, 1].plot(hours, recv_before, label="Receive buses: before")
    axes[0, 1].plot(hours, recv_after, label="Receive buses: after")
    axes[0, 1].set_title("(b) Geo-flex load on send/receive buses")
    axes[0, 1].set_xlabel("Hour")
    axes[0, 1].set_ylabel("MW")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # (c) Total shifted amount and backlog
    axes[1, 0].plot(hours, moved, label="Total shifted amount")
    axes[1, 0].plot(hours, backlog, label="Total backlog")
    axes[1, 0].set_title("(c) Shift magnitude and backlog")
    axes[1, 0].set_xlabel("Hour")
    axes[1, 0].set_ylabel("MW")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (d) Delayable load
    axes[1, 1].plot(hours, delay_arrival, label="Delayable arrival")
    axes[1, 1].plot(hours, delay_served, label="Delayable served")
    axes[1, 1].set_title("(d) Delayable data-center load")
    axes[1, 1].set_xlabel("Hour")
    axes[1, 1].set_ylabel("MW")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    if out_png:
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.show()

def print_geo_shifts(case_data: dict, vars_dict: dict, max_hours: int | None = None):
    baseMVA = case_data["baseMVA"]
    delta = np.asarray(vars_dict["delta"].value) * baseMVA
    bus_ids = np.asarray(case_data["bus_ids"])

    T = delta.shape[1]
    H = T if max_hours is None else min(T, max_hours)

    print("\n=== Geo-shifting vector by hour (MW) ===")
    for t in range(H):
        print(f"\nHour {t}")
        print("bus_id, net_delta")
        for i, bus_id in enumerate(bus_ids):
            dd = delta[i, t]
            if abs(dd) > 1e-6:
                print(f"{int(bus_id)}, {dd:.4f}")

def print_geo_shift_summary(case_data: dict, vars_dict: dict, flex_data: dict):
    baseMVA = case_data["baseMVA"]
    delta = np.asarray(vars_dict["delta"].value) * baseMVA
    send_idx = np.asarray(flex_data["send_bus_idx"], dtype=int)
    recv_idx = np.asarray(flex_data["recv_bus_idx"], dtype=int)
    bus_ids = np.asarray(case_data["bus_ids"])

    moved = 0.5 * np.sum(np.abs(delta), axis=0)

    print("\n=== Geo-shift summary ===")
    print(f"Average shifted amount per hour (MW): {np.mean(moved):.4f}")
    print(f"Maximum shifted amount in any hour (MW): {np.max(moved):.4f}")

    avg_abs_shift = np.mean(np.abs(delta), axis=1)
    top = np.argsort(-avg_abs_shift)[:15]
    print("\nTop buses by average absolute net shift (MW):")
    for i in top:
        print(f"bus {int(bus_ids[i])}: {avg_abs_shift[i]:.4f}")


def run_wecc240_capacity_expansion(cfg: SolveConfig, flex: FlexConfig):
    case = load_wecc_case(cfg.wecc_case_py)
    case_data = build_dc_network(case)
    hourly_df = read_wecc_hourly_load(cfg.load_csv, start_hour=cfg.start_hour, horizon_hours=cfg.horizon_hours)
    L_base = build_bus_load_matrix_from_total_profile(
        case, hourly_df, target="average", target_scale=1.9) / case_data["baseMVA"]
    baseMVA = case_data["baseMVA"]
    print("Peak baseline load (MW):", np.max(np.sum(L_base, axis=0)) * baseMVA)
    print("Average baseline load (MW):", np.mean(np.sum(L_base, axis=0)) * baseMVA)
    print("Total PMAX (MW):", np.sum(case_data["Pmax"]) * baseMVA)
    flex_data = build_flexible_data_center_loads(L_base, case_data["bus_ids"], flex)

    problem, vars_dict = build_wecc_cvxpy_model(case_data, L_base, flex_data, cfg)
    solver_used = solve_problem(problem, verbose=True)

    print("\nUsed solver:", solver_used)
    print("Objective:", problem.value)
    print("Added capacity total:", float(np.sum(vars_dict["x"].value)))
    if vars_dict["load_shed"] is not None:
        print("Total load shed:", float(np.sum(vars_dict["load_shed"].value)))
    #plot_results(case_data, vars_dict, out_png="wecc240_results.png")
    print_geo_shift_summary(case_data, vars_dict, flex_data)
    #print_geo_shifts(case_data, vars_dict)
    #plot_results(case_data, vars_dict, flex_data, out_png="wecc240_results.png")
    return problem, vars_dict, case_data, flex_data


if __name__ == "__main__":
    # Example bus IDs below are placeholders that exist in the WECC-240 case.
    # Replace them with the buses you want to host delayable / geo-flexible data-center load.
    if __name__ == "__main__":
        cfg = SolveConfig(
            load_csv="wecc_load.csv",
            wecc_case_py="wecc240_2011.py",
            horizon_hours=24 * 10,
            start_hour=0,
            delay_penalty=1.0,
            shift_penalty=0.0001,
            slack_penalty=None,
            expansion_cost_scale=200.0,
            expansion_limit_fraction=0.3,
        )

        case = load_wecc_case(cfg.wecc_case_py)
        case_data = build_dc_network(case)
        bus = case_data["bus"]
        pd_bus = bus[:, 2]
        bus_ids = case_data["bus_ids"]

        top_idx = np.argsort(-pd_bus)
        top_bus_ids = [bus_ids[i] for i in top_idx[:100]]

        print("Top load buses:", top_bus_ids)
        print("Top PD values:", pd_bus[top_idx[:10]])

        flex = FlexConfig(
            dc_delay_buses=top_bus_ids[:50],
            geo_send_buses=top_bus_ids[:30],
            geo_receive_buses=top_bus_ids[40:80],
            delay_share_of_system_load=0.2,
            geo_share_of_system_load=0.1,
            delay_window_hours=6,  # overwritten in sweep
            max_send_fraction=0.50,
            max_receive_fraction=0.50,
            shift_budget_fraction=0.50,
        )

        sweep_delay_window(cfg, flex, delay_windows=[0, 1, 2, 3, 4, 6], plot_bus_idx=0, plot_day=0)
