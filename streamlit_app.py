# app.py
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
import pandas as pd


# -----------------------------
# Core stats helpers
# -----------------------------

def se_diff(sd: float, n_per_group: int) -> float:
    return math.sqrt(2) * sd / math.sqrt(n_per_group)

def compute_alpha_beta_power(mde: float, sd: float, n_per_group: int, alpha: float = 0.05, two_sided: bool = True):
    """
    Returns critical values in raw units, beta, power, and SE for difference-in-means.
    Normal approximation.
    """
    se = se_diff(sd, n_per_group)

    if two_sided:
        z_crit = norm.ppf(1 - alpha / 2)
        crit_left = -z_crit * se
        crit_right = z_crit * se
        beta = norm.cdf(crit_right, loc=mde, scale=se) - norm.cdf(crit_left, loc=mde, scale=se)
    else:
        z_crit = norm.ppf(1 - alpha)
        crit_left = None
        crit_right = z_crit * se
        beta = norm.cdf(crit_right, loc=mde, scale=se)

    power = 1 - beta

    return {
        "se": se,
        "z_crit": z_crit,
        "crit_left": crit_left,
        "crit_right": crit_right,
        "alpha": alpha,
        "beta": beta,
        "power": power
    }

def fdr_tdr_from_rates(power: float, alpha: float, p_h1: float):
    """
    p_h1 is prevalence/base rate of true effects across experiments: π1 = P(H1 true)
    """
    pi1 = p_h1
    pi0 = 1 - p_h1
    fp_rate = pi0 * alpha
    tp_rate = pi1 * power
    denom = fp_rate + tp_rate
    if denom <= 0:
        return float("nan"), float("nan")
    fdr = fp_rate / denom
    tdr = tp_rate / denom
    return fdr, tdr

def outcome_shares(power: float, alpha: float, p_h1: float):
    """
    Unconditional shares across all experiments; they sum to 1.
    """
    pi1 = p_h1
    pi0 = 1 - p_h1
    TP = pi1 * power
    FN = pi1 * (1 - power)
    FP = pi0 * alpha
    TN = pi0 * (1 - alpha)
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

def conditional_given_fail(shares: dict):
    """
    Given unconditional shares TP/FP/TN/FN, compute P(H0|fail) and P(H1|fail),
    where "fail" == fail-to-reject (non-significant).
    """
    p_fail = shares["TN"] + shares["FN"]
    if p_fail <= 0:
        return float("nan"), float("nan")
    return shares["TN"] / p_fail, shares["FN"] / p_fail


# -----------------------------
# Plotting functions (return figures)
# -----------------------------

def fig_alpha_beta(mde: float, sd: float, n_per_group: int, alpha: float = 0.05, two_sided: bool = True, title: str = ""):
    stt = compute_alpha_beta_power(mde, sd, n_per_group, alpha, two_sided)
    se = stt["se"]
    crit_left = stt["crit_left"]
    crit_right = stt["crit_right"]
    beta = stt["beta"]
    power = stt["power"]

    x_min = min(-4 * se, mde - 4 * se)
    x_max = max(4 * se, mde + 4 * se)
    x = np.linspace(x_min, x_max, 1000)

    h0 = norm.pdf(x, loc=0, scale=se)
    h1 = norm.pdf(x, loc=mde, scale=se)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, h0, label="H₀: sampling dist (mean = 0)")
    ax.plot(x, h1, label=f"H₁: sampling dist (mean = {mde:g})")

    if two_sided:
        xl = np.linspace(x_min, crit_left, 300)
        xr = np.linspace(crit_right, x_max, 300)
        ax.fill_between(xl, 0, norm.pdf(xl, 0, se), alpha=0.25, label="Type I error (α)")
        ax.fill_between(xr, 0, norm.pdf(xr, 0, se), alpha=0.25)

        xn = np.linspace(crit_left, crit_right, 300)
        ax.fill_between(xn, 0, norm.pdf(xn, mde, se), alpha=0.25, label="Type II error (β)")

        ax.axvline(crit_left, linestyle="--")
        ax.axvline(crit_right, linestyle="--")
    else:
        xr = np.linspace(crit_right, x_max, 300)
        ax.fill_between(xr, 0, norm.pdf(xr, 0, se), alpha=0.25, label="Type I error (α)")

        xn = np.linspace(x_min, crit_right, 300)
        ax.fill_between(xn, 0, norm.pdf(xn, mde, se), alpha=0.25, label="Type II error (β)")
        ax.axvline(crit_right, linestyle="--")

    ax.text(0.02, 0.90, f"α = {alpha:.2%}", transform=ax.transAxes)
    ax.text(0.02, 0.84, f"β = {beta:.2%}", transform=ax.transAxes)
    ax.text(0.02, 0.77, f"Power = {power:.2%}", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.text(0.02, 0.70, f"SE(diff) = {se:.4g}", transform=ax.transAxes)

    ax.set_title(title or "Sampling distributions (raw diff-in-means units)")
    ax.set_xlabel("Observed difference in sample means")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    return fig, stt

def fig_outcomes_dashboard(mde: float, sd: float, n_per_group: int, alpha: float, two_sided: bool, p_h1: float):
    stt = compute_alpha_beta_power(mde, sd, n_per_group, alpha, two_sided)
    power = stt["power"]

    shares = outcome_shares(power, alpha, p_h1)
    fdr, tdr = fdr_tdr_from_rates(power, alpha, p_h1)
    p_h0_fail, p_h1_fail = conditional_given_fail(shares)

    p_reject = shares["TP"] + shares["FP"]
    p_fail = shares["TN"] + shares["FN"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (1) Unconditional shares
    ax = axes[0]
    labels = ["True discovery (TP)", "False discovery (FP)", "Correct null (TN)", "Missed discovery (FN)"]
    vals = [shares["TP"], shares["FP"], shares["TN"], shares["FN"]]
    bottom = 0.0
    for lab, val in zip(labels, vals):
        ax.bar([0], [val], bottom=bottom, label=f"{lab}: {val:.3f}")
        bottom += val
    ax.set_ylim(0, 1)
    ax.set_xticks([0])
    ax.set_xticklabels(["All experiments"])
    ax.set_title("Unconditional shares\n(sum to 1)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2, axis="y")

    # (2) Composition among rejects
    ax = axes[1]
    ax.bar(["Reject H0"], [tdr], label=f"TDR: {tdr:.2%}")
    ax.bar(["Reject H0"], [fdr], bottom=[tdr], label=f"FDR: {fdr:.2%}")
    ax.set_ylim(0, 1)
    ax.set_title("Composition of significant results")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")
    ax.text(0.02, 0.02, f"P(reject)={p_reject:.3f}", transform=ax.transAxes, fontsize=9)

    # (3) Composition among fails
    ax = axes[2]
    ax.bar(["Fail to Reject"], [p_h0_fail], label=f"P(H0|fail): {p_h0_fail:.2%}")
    ax.bar(["Fail to Reject"], [p_h1_fail], bottom=[p_h0_fail], label=f"P(H1|fail): {p_h1_fail:.2%}")
    ax.set_ylim(0, 1)
    ax.set_title("Composition of non-significant results")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")
    ax.text(0.02, 0.02, f"P(fail)={p_fail:.3f}", transform=ax.transAxes, fontsize=9)

    fig.suptitle(
        f"Discovery outcomes dashboard\n"
        f"MDE={mde}, SD={sd}, n={n_per_group}, α={alpha}, P(H1)={p_h1}, power={power:.2%}",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    return fig, {
        "stats": stt,
        "shares": shares,
        "fdr": fdr,
        "tdr": tdr,
        "p_reject": p_reject,
        "p_fail": p_fail,
        "p_h0_fail": p_h0_fail,
        "p_h1_fail": p_h1_fail
    }

def fig_fdr_vs_n(mde: float, sd: float, alpha: float, two_sided: bool, p_h1: float, n_max: int, n_marker: int):
    ns = np.arange(20, n_max + 1, 20)

    fdrs = []
    powers = []
    for n in ns:
        stt = compute_alpha_beta_power(mde, sd, n, alpha, two_sided)
        powers.append(stt["power"])
        fdrs.append(fdr_tdr_from_rates(stt["power"], alpha, p_h1)[0])

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(ns, fdrs, linewidth=2, label="FDR")
    ax1.set_xlabel("Sample size per group (n)")
    ax1.set_ylabel("FDR")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ns, powers, color="gray", linestyle="--", alpha=0.8, label="Power")
    ax2.set_ylabel("Power")
    ax2.set_ylim(0, 1)

    if n_marker is not None:
        stt_m = compute_alpha_beta_power(mde, sd, n_marker, alpha, two_sided)
        fdr_m, _ = fdr_tdr_from_rates(stt_m["power"], alpha, p_h1)
        ax1.scatter([n_marker], [fdr_m], zorder=5)
        ax1.annotate(
            f"n={n_marker}\nFDR={fdr_m:.2%}\npower={stt_m['power']:.2%}",
            xy=(n_marker, fdr_m),
            xytext=(min(n_max, int(n_marker * 1.25)), min(0.95, fdr_m + 0.12)),
            arrowprops=dict(arrowstyle="->"),
            fontsize=9
        )

    ax1.legend(loc="upper right")
    ax1.set_title(f"FDR vs n (MDE={mde}, SD={sd}, α={alpha}, P(H1)={p_h1})")

    return fig


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Alpha/Beta + FDR Teaching Dashboard", layout="wide")

st.title("Experimentation error rates: α/β/power and false discoveries")

tabs = st.tabs(["1) α/β distributions", "2) FDR outcomes + distributions"])

with st.sidebar:
    st.header("Global assumptions")
    n_per_group = st.number_input("Sample size per group (n)", min_value=10, max_value=5_000_000, value=4000, step=50)
    two_sided = st.checkbox("Two-sided test", value=True)
    alpha = st.slider("Alpha (α)", min_value=0.001, max_value=0.20, value=0.05, step=0.001)
    mde = st.number_input("MDE (effect under H1)", value=0.02, format="%.6f")
    sd = st.number_input("SD (per-user outcome)", value=0.20, format="%.6f")


# ---- Tab 1 ----
with tabs[0]:
    st.subheader("Sampling distributions and where α and β come from")

    fig1, stats1 = fig_alpha_beta(
        mde=mde, sd=sd, n_per_group=int(n_per_group), alpha=alpha, two_sided=two_sided,
        title="How α and β arise from sampling distributions"
    )
    st.pyplot(fig1, clear_figure=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Power", f"{stats1['power']:.2%}")
    c2.metric("Beta (β)", f"{stats1['beta']:.2%}")
    c3.metric("SE(diff)", f"{stats1['se']:.6g}")

    st.caption("Interpretation")
    st.write(
        "Dashed lines are rejection thresholds implied by α. "
        "Shaded tails under H0 are α. Shaded non-rejection region under H1 is β. "
        "Power = 1 − β."
    )


# ---- Tab 2 ----
with tabs[1]:
    st.subheader("FDR outcomes (and the underlying sampling distributions)")

    colL, colR = st.columns([1, 2])

    with colL:
        p_h1 = st.slider("P(H1 true) base rate (π1)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        n_max = st.slider("Max n for FDR curve", min_value=200, max_value=50_000, value=20_000, step=200)

        st.caption("Key point")
        st.write(
            "α is fixed by your threshold. Power changes with n/SD/MDE. "
            "FDR is the fraction of significant results that are actually null, "
            "and it depends on α, power, and the base rate P(H1)."
        )

    with colR:
        # Sampling distribution for current settings
        fig_sd, stats2 = fig_alpha_beta(
            mde=mde, sd=sd, n_per_group=int(n_per_group), alpha=alpha, two_sided=two_sided,
            title="Sampling distributions for current FDR settings"
        )
        st.pyplot(fig_sd, clear_figure=True)

        # Outcomes dashboard
        fig_dash, out = fig_outcomes_dashboard(
            mde=mde, sd=sd, n_per_group=int(n_per_group), alpha=alpha, two_sided=two_sided, p_h1=p_h1
        )
        st.pyplot(fig_dash, clear_figure=True)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Power", f"{out['stats']['power']:.2%}")
        m2.metric("FDR = P(H0 | reject)", f"{out['fdr']:.2%}")
        m3.metric("TDR = P(H1 | reject)", f"{out['tdr']:.2%}")
        m4.metric("P(reject)", f"{out['p_reject']:.2%}")

        # FDR curve (marker uses the same global n)
        fig_curve = fig_fdr_vs_n(
            mde=mde, sd=sd, alpha=alpha, two_sided=two_sided, p_h1=p_h1,
            n_max=int(n_max), n_marker=int(n_per_group)
        )
        st.pyplot(fig_curve, clear_figure=True)

        # 2x2 table for unconditional outcome shares
        shares = out["shares"]
        shares_table = pd.DataFrame(
            {
                "Reject H0": {
                    "H1 true": shares["TP"],  # True discovery share
                    "H0 true": shares["FP"],  # False discovery share
                },
                "Fail to Reject H0": {
                    "H1 true": shares["FN"],  # Missed discovery share
                    "H0 true": shares["TN"],  # Correct null share
                },
            }
        )

        st.write("Outcome shares across all experiments (sum to 1):")
        st.dataframe(shares_table.style.format("{:.4f}"))
