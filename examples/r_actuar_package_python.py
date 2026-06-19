"""Python conversion of the ruin-theory examples from ``R_actuar_package.pdf``.

Running this script creates:

- ``output/pdf/r_actuar_package_python.pdf``: a compact PDF report with tables
  and figures analogous to the R actuar examples;
- ``output/figures/*.png``: standalone graphs generated with the package's
  plotting helpers.

The code uses the public ``ruin_theory`` package for distributions, models,
closed forms, loss discretization, Panjer/Pollaczek-Khinchine approximations,
simulation and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from ruin_theory import (
    ClaimDistribution,
    CramerLundbergProcess,
    PreventionProgram,
    adjustment_coefficient,
    cramer_lundberg_asymptotic,
    discretize,
    discrete_pollaczek_khinchine_ultimate_ruin,
    estimate_ruin_probability,
    exponential,
    heavy_tail_integrated_tail_asymptotic,
    lundberg_bound,
    mixture_exponential,
    plot_path,
    plot_paths,
    plot_ruin_curve,
    plot_ruin_time_histogram,
    plot_terminal_reserve_distribution,
    simulate_path,
    simulate_terminal_reserves,
    ultimate_ruin_exponential,
    ultimate_ruin_hyperexponential,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
PDF_DIR = OUTPUT / "pdf"
PDF_PATH = PDF_DIR / "r_actuar_package_python.pdf"


@dataclass(frozen=True)
class ActuarTables:
    adjustment: float
    retention_alphas: np.ndarray
    retention_coefficients: np.ndarray
    surplus_grid: np.ndarray
    exponential_ruin: np.ndarray
    hyperexponential_ruin: np.ndarray
    beekman_grid: np.ndarray
    beekman_lower: np.ndarray
    beekman_upper: np.ndarray


def _ensure_output_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def _set_report_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfaf7",
            "axes.facecolor": "#fbfaf7",
            "axes.edgecolor": "#2f3640",
            "axes.labelcolor": "#2f3640",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 10,
            "grid.color": "#ded8cc",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def _polish_axis(ax) -> None:
    ax.grid(True, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, name: str) -> Path:
    path = FIGURES / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def _add_page_title(fig, title: str, subtitle: str | None = None) -> None:
    fig.text(0.06, 0.95, title, fontsize=18, fontweight="bold", color="#1f2933")
    if subtitle:
        fig.text(0.06, 0.91, subtitle, fontsize=10.5, color="#4b5563")


def _table_page(
    pdf: PdfPages,
    *,
    title: str,
    subtitle: str,
    columns: list[str],
    rows: list[list[str]],
    note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    _add_page_title(fig, title, subtitle)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.5)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d2cabd")
        if row == 0:
            cell.set_facecolor("#284b63")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f3efe7")
    if note:
        fig.text(0.06, 0.08, note, fontsize=9, color="#4b5563")
    pdf.savefig(fig)
    plt.close(fig)


def _lomax_claim(shape: float, scale: float) -> ClaimDistribution:
    """Lomax/Pareto-II law used by actuar's ``ppareto(x, shape, scale)``."""

    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return scale * rng.pareto(shape, size=n)

    def cdf(x):
        values = np.asarray(x, dtype=float)
        return np.where(values < 0.0, 0.0, 1.0 - (scale / (scale + values)) ** shape)

    def survival(x):
        values = np.asarray(x, dtype=float)
        return np.where(values < 0.0, 1.0, (scale / (scale + values)) ** shape)

    return ClaimDistribution(
        name="lomax",
        mean_value=scale / (shape - 1.0),
        variance_value=scale**2 * shape / ((shape - 1.0) ** 2 * (shape - 2.0)),
        sampler=sampler,
        cdf_function=cdf,
        survival_function=survival,
        metadata={"shape": float(shape), "scale": float(scale)},
    )


def compute_actuar_tables() -> ActuarTables:
    adjustment_model = CramerLundbergProcess(
        premium_rate=2.4,
        claim_arrival_rate=2.0,
        claim_distribution=exponential(rate=1.0),
    )
    adjustment = adjustment_coefficient(adjustment_model)

    alphas = np.array([0.75, 0.80, 0.90, 1.00])
    retention_coefficients = np.array(
        [
            adjustment_coefficient(
                CramerLundbergProcess(
                    premium_rate=2.6 * alpha - 0.2,
                    claim_arrival_rate=2.0,
                    claim_distribution=exponential(rate=1.0 / alpha),
                )
            )
            for alpha in alphas
        ]
    )

    u = np.arange(11)
    exponential_model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=exponential(rate=5.0),
    )
    hyper_model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=mixture_exponential(rates=[3.0, 7.0], weights=[0.5, 0.5]),
    )

    equilibrium = _lomax_claim(shape=4.0, scale=4.0)
    f_lower = discretize(equilibrium, from_=0.0, to=200.0, step=1.0, method="lower").pmf
    f_upper = discretize(equilibrium, from_=0.0, to=200.0, step=1.0, method="upper").pmf
    grid = np.arange(0, 55, 5)

    return ActuarTables(
        adjustment=adjustment,
        retention_alphas=alphas,
        retention_coefficients=retention_coefficients,
        surplus_grid=u,
        exponential_ruin=ultimate_ruin_exponential(exponential_model, u),
        hyperexponential_ruin=ultimate_ruin_hyperexponential(hyper_model, u),
        beekman_grid=grid,
        beekman_lower=discrete_pollaczek_khinchine_ultimate_ruin(
            f_upper,
            grid,
            step=1.0,
            rho=5.0 / 6.0,
        ),
        beekman_upper=discrete_pollaczek_khinchine_ultimate_ruin(
            f_lower,
            grid,
            step=1.0,
            rho=5.0 / 6.0,
        ),
    )


def make_adjustment_figure() -> tuple[plt.Figure, Path]:
    alphas = np.linspace(0.36, 1.0, 140)
    coefficients = np.array(
        [
            adjustment_coefficient(
                CramerLundbergProcess(
                    premium_rate=2.6 * alpha - 0.2,
                    claim_arrival_rate=2.0,
                    claim_distribution=exponential(rate=1.0 / alpha),
                )
            )
            for alpha in alphas
        ]
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(alphas, coefficients, color="#284b63", linewidth=2.2)
    ax.scatter(
        [0.75, 0.80, 0.90, 1.0],
        [0.19047619, 0.18617021, 0.17653167, 0.16666667],
        color="#b65c44",
    )
    ax.set_title("Adjustment coefficient under proportional reinsurance")
    ax.set_xlabel("retention alpha")
    ax.set_ylabel("adjustment coefficient")
    _polish_axis(ax)
    return fig, _save_figure(fig, "fig_adjustment_coefficients.png")


def make_ruin_curve_figure() -> tuple[plt.Figure, Path]:
    u = np.linspace(0.0, 10.0, 300)
    exponential_model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=exponential(rate=5.0),
    )
    hyper_model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=mixture_exponential(rates=[3.0, 7.0], weights=[0.5, 0.5]),
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plot_ruin_curve(
        u,
        ultimate_ruin_exponential(exponential_model, u),
        ax=ax,
        label="Exp(5) claims",
    )
    ax.lines[-1].set_color("#0b6e4f")
    ax.plot(
        u,
        ultimate_ruin_hyperexponential(hyper_model, u),
        color="#b65c44",
        linewidth=2.0,
        label="0.5 Exp(3) + 0.5 Exp(7)",
    )
    ax.plot(
        u,
        cramer_lundberg_asymptotic(hyper_model, u),
        color="#2f3640",
        linewidth=1.4,
        linestyle=":",
        label="hyperexponential asymptotic",
    )
    ax.set_title("Ultimate ruin probabilities")
    ax.legend()
    _polish_axis(ax)
    return fig, _save_figure(fig, "fig_ruin_curves.png")


def make_beekman_figure(tables: ActuarTables) -> tuple[plt.Figure, Path]:
    mid = 0.5 * (tables.beekman_lower + tables.beekman_upper)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plot_ruin_curve(
        tables.beekman_grid,
        mid,
        ax=ax,
        label="Beekman/Panjer midpoint",
        ci_low=tables.beekman_lower,
        ci_high=tables.beekman_upper,
        band_alpha=0.24,
    )
    ax.lines[-1].set_color("#284b63")
    ax.collections[-1].set_facecolor("#b65c44")
    ax.set_title("Beekman/Panjer bounds for the Pareto example")
    _polish_axis(ax)
    return fig, _save_figure(fig, "fig_beekman_panjer_bounds.png")


def make_simulation_diagnostics_figure() -> tuple[plt.Figure, Path]:
    model = CramerLundbergProcess(
        initial_capital=2.0,
        premium_rate=1.2,
        claim_arrival_rate=1.0,
        claim_distribution=exponential(rate=1.0),
        prevention=PreventionProgram(frequency_multiplier=0.92, severity_multiplier=0.9),
    )
    horizon = 10.0
    path = simulate_path(model, horizon=horizon, seed=11, stop_at_ruin=False)
    rng = np.random.default_rng(123)
    paths = [
        simulate_path(model, horizon=horizon, seed=rng, stop_at_ruin=False)
        for _ in range(35)
    ]
    estimate = estimate_ruin_probability(model, horizon=horizon, n_simulations=1200, seed=321)
    terminal = simulate_terminal_reserves(model, horizon=horizon, n_simulations=1200, seed=456)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)
    plot_path(path, ax=axes[0, 0], show_ruin=True)
    plot_paths(paths, ax=axes[0, 1], alpha=0.23)
    plot_ruin_time_histogram(estimate, ax=axes[1, 0], bins=24)
    plot_terminal_reserve_distribution(terminal, ax=axes[1, 1], bins=28)
    for axis in axes.ravel():
        _polish_axis(axis)
    fig.suptitle(
        "Simulation diagnostics generated by ruin_theory.plotting",
        fontsize=14,
        fontweight="bold",
    )
    return fig, _save_figure(fig, "fig_simulation_diagnostics.png")


def make_lundberg_figure() -> tuple[plt.Figure, Path]:
    model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=exponential(rate=5.0),
    )
    u = np.linspace(0.0, 6.0, 200)
    exact = ultimate_ruin_exponential(model, u)
    bound = lundberg_bound(model, u)
    asymptotic = cramer_lundberg_asymptotic(model, u)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plot_ruin_curve(u, exact, ax=ax, label="exact")
    ax.lines[-1].set_color("#0b6e4f")
    ax.plot(u, bound, color="#b65c44", linewidth=1.8, linestyle="--", label="Lundberg bound")
    ax.plot(u, asymptotic, color="#284b63", linewidth=1.4, linestyle=":", label="CL asymptotic")
    ax.set_title("Exact exponential ruin, Lundberg bound and asymptotic")
    ax.legend()
    _polish_axis(ax)
    return fig, _save_figure(fig, "fig_lundberg_bound.png")


def make_heavy_tail_figure() -> tuple[plt.Figure, Path]:
    model = CramerLundbergProcess(
        premium_rate=1.2,
        claim_arrival_rate=1.0,
        claim_distribution=_lomax_claim(shape=5.0, scale=4.0),
    )
    u = np.linspace(5.0, 100.0, 240)
    equilibrium_tail = lambda x: (4.0 / (4.0 + np.asarray(x, dtype=float))) ** 4
    approximation = heavy_tail_integrated_tail_asymptotic(
        model,
        u,
        integrated_tail_survival=equilibrium_tail,
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(u, approximation, color="#284b63", linewidth=2.0)
    ax.set_yscale("log")
    ax.set_title("Heavy-tail integrated-tail asymptotic for the actuar Pareto example")
    ax.set_xlabel("initial surplus")
    ax.set_ylabel("asymptotic ruin probability")
    _polish_axis(ax)
    return fig, _save_figure(fig, "fig_heavy_tail_asymptotic.png")


def build_pdf(tables: ActuarTables, figures: list[tuple[plt.Figure, Path]]) -> None:
    with PdfPages(PDF_PATH) as pdf:
        cover = plt.figure(figsize=(11, 8.5))
        cover.patch.set_facecolor("#fbfaf7")
        _add_page_title(
            cover,
            "Python conversion of R actuar ruin examples",
            "Generated with the local ruin_theory package",
        )
        cover.text(
            0.06,
            0.80,
            "\n".join(
                [
                    "This report mirrors the ruin-theory code examples from R_actuar_package.pdf.",
                    "It uses ruin_theory for Cramer-Lundberg models, exact formulas,",
                    "loss discretization, Panjer/Pollaczek-Khinchine approximations,",
                    "Monte Carlo simulation, and plotting diagnostics.",
                    "",
                    "Generated artifacts:",
                    f"- {PDF_PATH.relative_to(ROOT)}",
                    "- output/figures/*.png",
                ]
            ),
            fontsize=12,
            color="#2f3640",
            va="top",
        )
        cover.text(
            0.06,
            0.16,
            "Note: the Beekman/Panjer bounds are produced with the package-level\n"
            "discretize and discrete Pollaczek-Khinchine routines.",
            fontsize=10,
            color="#4b5563",
        )
        pdf.savefig(cover)
        plt.close(cover)

        _table_page(
            pdf,
            title="Adjustment coefficient examples",
            subtitle="R actuar adjCoef examples translated to CramerLundbergProcess models.",
            columns=["quantity", "value"],
            rows=[
                ["base adjustment coefficient", f"{tables.adjustment:.4f}"],
                *[
                    [f"retention alpha={alpha:.2f}", f"{coef:.4f}"]
                    for alpha, coef in zip(
                        tables.retention_alphas,
                        tables.retention_coefficients,
                        strict=True,
                    )
                ],
            ],
            note="The base example matches actuar's 0.1667 output.",
        )

        adjustment_fig, _ = figures[0]
        pdf.savefig(adjustment_fig)

        _table_page(
            pdf,
            title="Ruin probabilities from actuar::ruin examples",
            subtitle="Exponential/exponential and hyperexponential/exponential models.",
            columns=["u", "Exp(5) claims", "0.5 Exp(3)+0.5 Exp(7)"],
            rows=[
                [str(int(u)), f"{exp_value:.6g}", f"{hyper_value:.6g}"]
                for u, exp_value, hyper_value in zip(
                    tables.surplus_grid,
                    tables.exponential_ruin,
                    tables.hyperexponential_ruin,
                    strict=True,
                )
            ],
            note="The hyperexponential values match (24 exp(-u) + exp(-6u)) / 35.",
        )

        for fig, _path in figures[1:]:
            pdf.savefig(fig)

        _table_page(
            pdf,
            title="Beekman/Panjer bounds",
            subtitle="The Pareto example from R_actuar_package.pdf.",
            columns=["u", "lower", "upper"],
            rows=[
                [str(int(u)), f"{lower:.7f}", f"{upper:.5f}"]
                for u, lower, upper in zip(
                    tables.beekman_grid,
                    tables.beekman_lower,
                    tables.beekman_upper,
                    strict=True,
                )
            ],
            note="These are the same lower and upper bounds printed in the actuar note.",
        )

        notes = plt.figure(figsize=(11, 8.5))
        notes.patch.set_facecolor("#fbfaf7")
        _add_page_title(
            notes,
            "Conversion notes",
            "What is implemented directly in ruin_theory today.",
        )
        notes.text(
            0.06,
            0.82,
            "\n".join(
                [
                    "Implemented through public package APIs:",
                    "- exponential and hyperexponential ultimate ruin formulas;",
                    "- adjustment coefficients and Lundberg bounds;",
                    "- severity discretization and compound-geometric Panjer recursion;",
                    "- heavy-tail integrated-tail asymptotics;",
                    "- phase-type distributions and matrix-exponential ruin formulas;",
                    "- INAR/BINAR dependent by-claim simulation and ruin estimates;",
                    "- trajectory, ruin-time, terminal-reserve and ruin-curve plotting.",
                    "",
                    "Planned public APIs:",
                    "- Gerber-Shiu penalties and deficit/surplus-before-ruin diagnostics.",
                ]
            ),
            fontsize=11.5,
            color="#2f3640",
            va="top",
        )
        pdf.savefig(notes)
        plt.close(notes)


def main() -> None:
    _ensure_output_dirs()
    _set_report_style()
    tables = compute_actuar_tables()
    figures = [
        make_adjustment_figure(),
        make_ruin_curve_figure(),
        make_lundberg_figure(),
        make_beekman_figure(tables),
        make_heavy_tail_figure(),
        make_simulation_diagnostics_figure(),
    ]
    build_pdf(tables, figures)
    for fig, _path in figures:
        plt.close(fig)

    print(f"PDF written to {PDF_PATH}")
    print("Figures written to:")
    for _fig, path in figures:
        print(f"- {path}")


if __name__ == "__main__":
    main()
