import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from pypower.case14 import case14

plt.rcParams.update({
    'font.size': 14,          # base font size
    'axes.titlesize': 15,     # title
    'axes.labelsize': 14,     # x/y labels
    'xtick.labelsize': 12,    # x tick labels
    'ytick.labelsize': 12,    # y tick labels
    'legend.fontsize': 14,    # legend
    'figure.titlesize': 16    # figure title
})
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


def select_dc_buses(case_data, n_dc_buses=4, seed=3):
    rng = np.random.default_rng(seed)
    positive_load_idx = np.where(case_data["bus"][:, PD] > 0)[0]
    dc_idx = np.sort(rng.choice(positive_load_idx, size=n_dc_buses, replace=False))
    return dc_idx


def build_geo_flex_data(case_data, L_base, dc_idx,
                        dc_total_share=0.5,
                        flex_share=0.5,
                        recv_headroom=1.0,
                        shift_budget_fraction=1.0):
    """
    DC load is ADDITIONAL load on top of nominal load.
    flex_share is the fraction of total DC load that is geographically shiftable.
    """
    N, T = L_base.shape
    dc_idx = np.array(dc_idx, dtype=int)
    n_dc_buses = len(dc_idx)

    n_send = max(1, n_dc_buses // 2)
    send_idx = dc_idx[:n_send]
    recv_idx = dc_idx[n_send:]
    if len(recv_idx) == 0:
        recv_idx = dc_idx[-1:]

    # Additional total DC load
    L_dc_total = np.zeros_like(L_base)
    L_dc_total[dc_idx, :] = dc_total_share * L_base[dc_idx, :]

    # Flexible portion of DC load
    L_dc_flex = np.zeros_like(L_base)
    L_dc_flex[dc_idx, :] = flex_share * L_dc_total[dc_idx, :]

    delta_lower = np.zeros_like(L_base)
    delta_upper = np.zeros_like(L_base)

    # Send buses can shift out only the flexible DC portion
    delta_lower[send_idx, :] = -L_dc_flex[send_idx, :]
    delta_upper[send_idx, :] = 0.0

    # Receive buses can absorb shifted flexible load
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
        "dc_total_share": dc_total_share,
        "flex_share": flex_share,
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

    L_dc_total = geo_data["L_dc_total"]
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

    # Total grid load = nominal load + additional DC load + geographic shift
    for t in range(T):
        net_inj = Gmap @ p[:, t] - (L_base[:, t] + L_dc_total[:, t] + delta[:, t])
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


def compute_congestion_stats(case_data, vars_dict, line_scale=0.05, tol=1e-4):
    f = np.asarray(vars_dict["f"].value)          # p.u.
    Fmax = line_scale * case_data["Fmax"]        # p.u.

    congested_mask = np.abs(f) >= (Fmax[:, None] - tol)
    congested_hours_per_line = np.sum(congested_mask, axis=1)
    avg_num_congested_lines = float(np.mean(np.sum(congested_mask, axis=0)))
    max_num_congested_lines = int(np.max(np.sum(congested_mask, axis=0)))

    return {
        "congested_hours_per_line": congested_hours_per_line,
        "avg_num_congested_lines": avg_num_congested_lines,
        "max_num_congested_lines": max_num_congested_lines,
    }


def plot_growth_results(results, x_key, x_label, title_prefix):
    xvals = [r[x_key] for r in results]
    avg_cong = [r["avg_num_congested_lines"] for r in results]
    invest_cost = [r["invest_cost"] for r in results]
    added_capacity = [r["added_capacity_mw"] for r in results]
    avg_shift = [r["avg_shift_mw"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # (a) average congestion count
    axes[0, 0].plot(xvals, avg_cong, marker="*", color='red')
    axes[0, 0].set_title("(a) Average number of congested lines")
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].set_ylabel("Average congested lines / hour")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(xvals, invest_cost, marker="*", color='red')
    axes[0, 1].set_title("(b) Investment cost")
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].set_ylabel("Investment cost")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(xvals, added_capacity, marker="*", color='red')
    axes[1, 0].set_title("(c) Added capacity")
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel("Added capacity (MW)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(xvals, avg_shift, marker="*", color='red')
    axes[1, 1].set_title("(d) Average shifted load")
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].set_ylabel("Average shifted load (MW)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title_prefix, fontsize=14)
    plt.show()

def plot_growth_results_grouped(dc_total_share_grid, results_flex0, results_flex06):
    x = np.arange(len(dc_total_share_grid))
    width = 0.35

    avg_cong_0 = [r["avg_num_congested_lines"] for r in results_flex0]
    avg_cong_06 = [r["avg_num_congested_lines"] for r in results_flex06]

    invest_0 = [r["invest_cost"] for r in results_flex0]
    invest_06 = [r["invest_cost"] for r in results_flex06]

    added_0 = [r["added_capacity_mw"] for r in results_flex0]
    added_06 = [r["added_capacity_mw"] for r in results_flex06]

    op_0 = [r["op_cost"] for r in results_flex0]
    op_06 = [r["op_cost"] for r in results_flex06]

    labels = [f"{v:.1f}" for v in dc_total_share_grid]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # (a) average congestion count
    axes[0, 0].bar(x - width/2, avg_cong_0, width, label="flex_share = 0.0")
    axes[0, 0].bar(x + width/2, avg_cong_06, width, label="flex_share = 0.6")
    axes[0, 0].set_title("(a) Average number of congested lines")
    axes[0, 0].set_xlabel("DC load growth ratio")
    axes[0, 0].set_ylabel("Average congested lines / hour")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    # (b) investment cost
    axes[0, 1].bar(x - width/2, invest_0, width, label="flex_share = 0.0")
    axes[0, 1].bar(x + width/2, invest_06, width, label="flex_share = 0.6")
    axes[0, 1].set_title("(b) Investment cost")
    axes[0, 1].set_xlabel("DC load growth ratio")
    axes[0, 1].set_ylabel("Investment cost")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].legend()
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    # (c) added capacity
    axes[1, 0].bar(x - width/2, added_0, width, label="flex_share = 0.0")
    axes[1, 0].bar(x + width/2, added_06, width, label="flex_share = 0.6")
    axes[1, 0].set_title("(c) Added capacity")
    axes[1, 0].set_xlabel("DC load growth ratio")
    axes[1, 0].set_ylabel("Added capacity (MW)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    # (d) operation cost
    axes[1, 1].bar(x - width/2, op_0, width, label="flex_share = 0.0")
    axes[1, 1].bar(x + width/2, op_06, width, label="flex_share = 0.6")
    axes[1, 1].set_title("(d) Operation cost")
    axes[1, 1].set_xlabel("DC load growth ratio")
    axes[1, 1].set_ylabel("Operation cost")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.show()


def sweep_dc_growth():
    """
    Group 1: grow total DC load, and compare no flexibility vs moderate flexibility.
    """
    case_data = load_case14()
    baseMVA = case_data["baseMVA"]
    L_base = build_30day_load_profile(case_data, days=30, seed=2)

    dc_idx = select_dc_buses(case_data, n_dc_buses=4, seed=3)
    print("Selected DC buses:", case_data["bus_ids"][dc_idx])

    dc_total_share_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    val_flex=0.4
    flex_compare = [0.0, val_flex]

    line_scale = 0.01
    recv_headroom = 1.2
    shift_budget_fraction = 1.0
    expansion_limit_fraction = 0.8
    expansion_cost_scale = 200.0
    shift_penalty = 0.0001

    results_by_flex = {0.0: [], val_flex: []}

    print("\n=== Sweep: DC load growth, comparing flex_share = 0.0 vs 0.6 ===")
    for dc_total_share in dc_total_share_grid:
        print(f"\n--- dc_total_share = {dc_total_share:.2f} ---")
        for fixed_flex_share in flex_compare:
            geo_data = build_geo_flex_data(
                case_data,
                L_base,
                dc_idx=dc_idx,
                dc_total_share=dc_total_share,
                flex_share=fixed_flex_share,
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

            cong_stats = compute_congestion_stats(case_data, vars_dict, line_scale=line_scale)

            row = {
                "dc_total_share": dc_total_share,
                "flex_share": fixed_flex_share,
                "solver": solver,
                "total_cost": total_cost,
                "invest_cost": invest_cost,
                "op_cost": op_cost,
                "shift_cost": shift_cost,
                "added_capacity_mw": added_capacity_mw,
                "avg_shift_mw": avg_shift_mw,
                "avg_num_congested_lines": cong_stats["avg_num_congested_lines"],
                "max_num_congested_lines": cong_stats["max_num_congested_lines"],
                "congested_hours_per_line": cong_stats["congested_hours_per_line"],
            }
            results_by_flex[fixed_flex_share].append(row)

            print(
                f"flex_share={fixed_flex_share:>4.2f} | "
                f"total_cost={total_cost:>12.4f} | "
                f"invest_cost={invest_cost:>12.4f} | "
                f"op_cost={op_cost:>12.4f} | "
                f"shift_cost={shift_cost:>10.4f} | "
                f"added_cap={added_capacity_mw:>9.3f} MW | "
                f"avg_shift={avg_shift_mw:>8.3f} MW | "
                f"avg_cong_lines={cong_stats['avg_num_congested_lines']:>6.3f}"
            )

    plot_growth_results_grouped(
        dc_total_share_grid,
        results_by_flex[0.0],
        results_by_flex[val_flex],
    )
    return results_by_flex


def sweep_flex_growth():
    """
    Group 2: keep total DC load fixed, grow flexibility.
    """
    case_data = load_case14()
    baseMVA = case_data["baseMVA"]
    L_base = build_30day_load_profile(case_data, days=30, seed=2)

    dc_idx = select_dc_buses(case_data, n_dc_buses=6, seed=3)
    print("Selected DC buses:", case_data["bus_ids"][dc_idx])

    fixed_dc_total_share = 0.4
    flex_share_grid = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    line_scale = 0.015
    recv_headroom = 1.2
    shift_budget_fraction = 1.0
    expansion_limit_fraction = 1.0
    expansion_cost_scale = 40.0
    shift_penalty = 0.001

    results = []

    print("\n=== Sweep: DC flexibility growth ===")
    for flex_share in flex_share_grid:
        geo_data = build_geo_flex_data(
            case_data,
            L_base,
            dc_idx=dc_idx,
            dc_total_share=fixed_dc_total_share,
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

        cong_stats = compute_congestion_stats(case_data, vars_dict, line_scale=line_scale)

        row = {
            "dc_total_share": fixed_dc_total_share,
            "flex_share": flex_share,
            "solver": solver,
            "total_cost": total_cost,
            "invest_cost": invest_cost,
            "op_cost": op_cost,
            "shift_cost": shift_cost,
            "added_capacity_mw": added_capacity_mw,
            "avg_shift_mw": avg_shift_mw,
            "avg_num_congested_lines": cong_stats["avg_num_congested_lines"],
            "max_num_congested_lines": cong_stats["max_num_congested_lines"],
            "congested_hours_per_line": cong_stats["congested_hours_per_line"],
        }
        results.append(row)

        print(
            f"dc_total_share={fixed_dc_total_share:>4.2f} | "
            f"flex_share={flex_share:>4.2f} | "
            f"invest_cost={invest_cost:>12.4f} | "
            f"added_cap={added_capacity_mw:>9.3f} MW | "
            f"avg_shift={avg_shift_mw:>8.3f} MW | "
            f"avg_cong_lines={cong_stats['avg_num_congested_lines']:>6.3f}"
            f"Total cost={total_cost:>12.3f} | "
        )

    plot_growth_results(
        results,
        x_key="flex_share",
        x_label="Flexible proportion of DC load",
        title_prefix="Group 2: Impact of DC flexibility growth"
    )
    return results


if __name__ == "__main__":
    results_dc_growth = sweep_dc_growth()
    results_flex_growth = sweep_flex_growth()