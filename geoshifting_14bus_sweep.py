import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from pypower.case14 import case14


# -----------------------------
# MATPOWER column indices
# -----------------------------
BUS_I = 0
PD = 2

F_BUS = 0
T_BUS = 1
BR_X = 3
RATE_A = 5
TAP = 8
BR_STATUS = 10

GEN_BUS = 0
GEN_STATUS = 7
PMAX = 8
PMIN = 9

MODEL = 0
NCOST = 3
COST = 4


def load_case14():
    ppc = case14()
    baseMVA = float(ppc["baseMVA"])
    bus = ppc["bus"].astype(float)
    branch = ppc["branch"].astype(float)
    gen = ppc["gen"].astype(float)
    gencost = ppc["gencost"].astype(float)

    active_branch = branch[branch[:, BR_STATUS] > 0.0]
    active_gen = gen[gen[:, GEN_STATUS] > 0.0]

    bus_ids = bus[:, BUS_I].astype(int)
    bus_map = {bus_id: i for i, bus_id in enumerate(bus_ids)}

    N = len(bus_ids)
    L = active_branch.shape[0]
    G = active_gen.shape[0]

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
        raw_rate = row[RATE_A]
        Fmax[ell] = (raw_rate / baseMVA) if raw_rate > 0 else 2.0

    Gmap = np.zeros((N, G))
    Pmax = np.zeros(G)
    Pmin = np.zeros(G)
    op_cost = np.zeros(G)

    for g, row in enumerate(active_gen):
        i = bus_map[int(row[GEN_BUS])]
        Gmap[i, g] = 1.0
        Pmax[g] = row[PMAX] / baseMVA
        Pmin[g] = max(row[PMIN], 0.0) / baseMVA

        if int(gencost[g, MODEL]) == 2:
            ncost = int(gencost[g, NCOST])
            coeffs = gencost[g, COST:COST + ncost]
            op_cost[g] = coeffs[-2] if ncost >= 2 else coeffs[-1]
        else:
            op_cost[g] = 1.0

    return {
        "baseMVA": baseMVA,
        "bus": bus,
        "branch": active_branch,
        "gen": active_gen,
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
        "N": N,
        "L": L,
        "G": G,
    }


def build_30day_load_profile(case_data, days=30, seed=1):
    rng = np.random.default_rng(seed)
    T = 24 * days

    base_pd = np.maximum(case_data["bus"][:, PD], 0.0) / case_data["baseMVA"]
    hours = np.arange(T)

    daily = 1.0 + 0.18 * np.sin(2 * np.pi * hours / 24.0 - 0.8)
    weekly = 1.0 + 0.06 * np.sin(2 * np.pi * hours / (24.0 * 7))
    noise = 0.02 * rng.standard_normal(T)
    profile = np.maximum(0.75, daily * weekly + noise)

    L_base = base_pd[:, None] * profile[None, :]
    return L_base


def build_fixed_dc_load(case_data, L_base, dc_idx, dc_total_share=0.6):
    """
    Fixed nodal DC load, same across all flexibility settings.
    """
    L_dc_total = np.zeros_like(L_base)
    L_dc_total[dc_idx, :] = dc_total_share * L_base[dc_idx, :]
    return L_dc_total


def build_geo_flex_from_fixed_dc(case_data, L_dc_total, dc_idx, flex_share,
                                 recv_headroom=1.0, shift_budget_fraction=1.0):
    """
    Keep total nodal DC load fixed, vary only the flexible proportion.
    """
    N, T = L_dc_total.shape
    dc_idx = np.array(dc_idx, dtype=int)
    n_dc_buses = len(dc_idx)

    n_send = max(1, n_dc_buses // 2)
    send_idx = dc_idx[:n_send]
    recv_idx = dc_idx[n_send:]
    if len(recv_idx) == 0:
        recv_idx = dc_idx[-1:]

    # Only this part is flexible
    L_dc_flex = np.zeros_like(L_dc_total)
    L_dc_flex[dc_idx, :] = flex_share * L_dc_total[dc_idx, :]

    delta_lower = np.zeros_like(L_dc_total)
    delta_upper = np.zeros_like(L_dc_total)

    # Send buses can reduce only the flexible portion
    delta_lower[send_idx, :] = -L_dc_flex[send_idx, :]
    delta_upper[send_idx, :] = 0.0

    # Receive buses can absorb flexible portion
    recv_cap_total = recv_headroom * np.sum(L_dc_flex[send_idx, :], axis=0)
    for bi in recv_idx:
        delta_lower[bi, :] = 0.0
        delta_upper[bi, :] = recv_cap_total / len(recv_idx)

    Gamma = shift_budget_fraction * np.sum(np.maximum(delta_upper, 0.0), axis=0)

    return {
        "L_dc_total": L_dc_total,
        "L_dc_flex": L_dc_flex,
        "delta_lower": delta_lower,
        "delta_upper": delta_upper,
        "Gamma": Gamma,
        "dc_idx": dc_idx,
        "send_idx": send_idx,
        "recv_idx": recv_idx,
        "dc_bus_ids": case_data["bus_ids"][dc_idx],
    }


def build_model(case_data, L_base, geo_data,
                line_scale=0.05,
                expansion_limit_fraction=0.5,
                expansion_cost_scale=30.0,
                shift_penalty=0.01):
    N, T = L_base.shape
    G = case_data["G"]
    L = case_data["L"]

    A_line = case_data["A_line"]
    B_line = case_data["B_line"]
    K = case_data["K"]
    Gmap = case_data["Gmap"]
    Pmax = case_data["Pmax"]
    Pmin = case_data["Pmin"]
    op_cost = case_data["op_cost"]
    Fmax = line_scale * case_data["Fmax"]

    delta_lower = geo_data["delta_lower"]
    delta_upper = geo_data["delta_upper"]
    Gamma = geo_data["Gamma"]

    Xmax = expansion_limit_fraction * Pmax
    capex = expansion_cost_scale * np.ones(G)

    x = cp.Variable(G, nonneg=True)
    p = cp.Variable((G, T))
    theta = cp.Variable((N, T))
    f = cp.Variable((L, T))

    delta = cp.Variable((N, T))
    z_shift = cp.Variable((N, T), nonneg=True)

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

    cons += [delta <= delta_upper, delta >= delta_lower]
    cons += [z_shift >= delta, z_shift >= -delta]

    for t in range(T):
        cons += [cp.sum(delta[:, t]) == 0.0]
        cons += [cp.sum(z_shift[:, t]) <= Gamma[t]]

    # Total load is unchanged except flexible DC part is geographically reallocated
    for t in range(T):
        net_inj = Gmap @ p[:, t] - (L_base[:, t] + delta[:, t])
        cons += [net_inj == K @ f[:, t]]

    invest_cost_expr = capex @ x
    op_cost_expr = cp.sum(cp.multiply(op_cost[:, None], p))
    shift_cost_expr = shift_penalty * cp.sum(z_shift)
    obj = invest_cost_expr + op_cost_expr + shift_cost_expr

    problem = cp.Problem(cp.Minimize(obj), cons)
    vars_dict = {
        "x": x,
        "p": p,
        "theta": theta,
        "f": f,
        "delta": delta,
        "z_shift": z_shift,
        "invest_cost_expr": invest_cost_expr,
        "op_cost_expr": op_cost_expr,
        "shift_cost_expr": shift_cost_expr,
    }
    return problem, vars_dict


def solve_problem(problem, verbose=False):
    installed = cp.installed_solvers()
    candidates = ["CLARABEL", "ECOS", "SCS", "SCIPY", "OSQP"]

    last_err = None
    for s in candidates:
        if s in installed:
            try:
                if s == "SCS":
                    problem.solve(solver=s, verbose=verbose, eps=1e-4, max_iters=30000)
                elif s == "OSQP":
                    problem.solve(solver=s, verbose=verbose, eps_abs=1e-5, eps_rel=1e-5)
                else:
                    problem.solve(solver=s, verbose=verbose)
                return s
            except Exception as e:
                last_err = e
    raise RuntimeError(f"All attempted solvers failed. Last error: {last_err}")


def plot_sweep_results(results):
    flex_share = [r["flex_share"] for r in results]
    total_cost = [r["total_cost"] for r in results]
    invest_cost = [r["invest_cost"] for r in results]
    added_capacity = [r["added_capacity_mw"] for r in results]
    avg_shift = [r["avg_shift_mw"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(flex_share, total_cost, marker="o")
    axes[0, 0].set_title("(a) Total cost vs flexible DC proportion")
    axes[0, 0].set_xlabel("Flexible proportion of fixed DC load")
    axes[0, 0].set_ylabel("Total cost")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(flex_share, invest_cost, marker="o")
    axes[0, 1].set_title("(b) Investment cost vs flexible DC proportion")
    axes[0, 1].set_xlabel("Flexible proportion of fixed DC load")
    axes[0, 1].set_ylabel("Investment cost")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(flex_share, added_capacity, marker="o")
    axes[1, 0].set_title("(c) Added capacity vs flexible DC proportion")
    axes[1, 0].set_xlabel("Flexible proportion of fixed DC load")
    axes[1, 0].set_ylabel("Added capacity (MW)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(flex_share, avg_shift, marker="o")
    axes[1, 1].set_title("(d) Average shifted load vs flexible DC proportion")
    axes[1, 1].set_xlabel("Flexible proportion of fixed DC load")
    axes[1, 1].set_ylabel("Average shifted load (MW)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.show()


def sweep_flex_share():
    case_data = load_case14()
    baseMVA = case_data["baseMVA"]
    L_base = build_30day_load_profile(case_data, days=30, seed=2)

    # Fix selected DC buses once for all runs
    rng = np.random.default_rng(3)
    positive_load_idx = np.where(case_data["bus"][:, PD] > 0)[0]
    dc_idx = np.sort(rng.choice(positive_load_idx, size=4, replace=False))

    print("Selected DC buses:", case_data["bus_ids"][dc_idx])

    # Fixed nodal DC load
    dc_total_share = 0.6
    L_dc_total = build_fixed_dc_load(case_data, L_base, dc_idx, dc_total_share=dc_total_share)

    flex_share_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    line_scale = 0.01
    recv_headroom = 1.2
    shift_budget_fraction = 1.0
    expansion_limit_fraction = 0.5
    expansion_cost_scale = 40.0
    shift_penalty = 0.001

    results = []

    print("\n=== Sweep results ===")
    print(f"Fixed DC total share at selected buses = {dc_total_share:.2f}")

    for flex_share in flex_share_grid:
        geo_data = build_geo_flex_from_fixed_dc(
            case_data,
            L_dc_total,
            dc_idx=dc_idx,
            flex_share=flex_share,
            recv_headroom=recv_headroom,
            shift_budget_fraction=shift_budget_fraction,
        )

        problem, vars_dict = build_model(
            case_data,
            L_base,
            geo_data,
            line_scale=line_scale,
            expansion_limit_fraction=expansion_limit_fraction,
            expansion_cost_scale=expansion_cost_scale,
            shift_penalty=shift_penalty,
        )

        solver = solve_problem(problem, verbose=False)

        x = np.asarray(vars_dict["x"].value).ravel()
        delta = np.asarray(vars_dict["delta"].value)

        invest_cost = float(vars_dict["invest_cost_expr"].value)
        op_cost = float(vars_dict["op_cost_expr"].value)
        shift_cost = float(vars_dict["shift_cost_expr"].value)
        total_cost = float(problem.value)

        added_capacity_pu = float(np.sum(x))
        added_capacity_mw = added_capacity_pu * baseMVA
        avg_shift_pu = float(np.mean(0.5 * np.sum(np.abs(delta), axis=0)))
        avg_shift_mw = avg_shift_pu * baseMVA

        row = {
            "flex_share": flex_share,
            "solver": solver,
            "total_cost": total_cost,
            "invest_cost": invest_cost,
            "op_cost": op_cost,
            "shift_cost": shift_cost,
            "added_capacity_pu": added_capacity_pu,
            "added_capacity_mw": added_capacity_mw,
            "avg_shift_pu": avg_shift_pu,
            "avg_shift_mw": avg_shift_mw,
        }
        results.append(row)

        print(
            f"flex_share={flex_share:>4.2f} | "
            f"total_cost={total_cost:>12.4f} | "
            f"invest_cost={invest_cost:>12.4f} | "
            f"shift_cost={shift_cost:>10.4f} | "
            f"added_cap={added_capacity_mw:>9.3f} MW | "
            f"avg_shift={avg_shift_mw:>8.3f} MW"
        )

    plot_sweep_results(results)
    return results


if __name__ == "__main__":
    sweep_flex_share()