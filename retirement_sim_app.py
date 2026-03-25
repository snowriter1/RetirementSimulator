"""
Retirement Portfolio Simulator — Streamlit App
================================================
Run with:
    streamlit run retirement_sim_app.py

    testing stuff

Dependencies:
    pip install streamlit numpy matplotlib scipy
"""

import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from scipy.stats import norm as sp_norm
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

BLUE   = '#378ADD'
GREEN  = '#1D9E75'
CORAL  = '#D85A30'
PURPLE = '#7F77DD'
AMBER  = '#BA7517'
LGRAY  = '#B4B2A9'

IRS_UNIFORM_LIFETIME = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0,
    79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8,
    85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2,
    91: 11.5, 92: 10.8, 93: 10.1, 94:  9.5, 95:  8.9, 96:  8.4,
    97:  7.8, 98:  7.3, 99:  6.8, 100: 6.4, 101: 6.0, 102: 5.6,
    103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9,
    109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0,
    115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def fmt_dollars(v):
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.0f}k"
    return f"${v:.0f}"

dollar_formatter = matplotlib.ticker.FuncFormatter(
    lambda v, _: fmt_dollars(v)
)

def pct(arr, q, axis=0):
    return np.percentile(arr, q, axis=axis)

def get_rmd(balance, age):
    """Return annual RMD amount. Zero if age < 73."""
    if age < 73:
        return 0.0
    period = IRS_UNIFORM_LIFETIME.get(int(age), 2.0)
    return balance / period

# ══════════════════════════════════════════════════════════════════════════════
# Bimodal sampler
# ══════════════════════════════════════════════════════════════════════════════

def make_bimodal_sampler(amp, sep, sigma_frac, bias_frac, pos_wt, mom, rng):
    """
    Stateful closure yielding one monthly dither value (%/month) per call.
    Mode switches with probability (1 - momentum) each month.
    """
    mu_pos =  sep * amp + bias_frac * amp
    mu_neg = -sep * amp + bias_frac * amp
    sigma  =  sigma_frac * amp
    mode   = 1 if rng.random() < pos_wt else -1

    def sample():
        nonlocal mode
        if rng.random() > mom:
            mode = 1 if rng.random() < pos_wt else -1
        mu = mu_pos if mode == 1 else mu_neg
        v = mu  # safe fallback if all rejection attempts fail
        for _ in range(30):
            v = rng.normal(mu, sigma)
            if abs(v) <= amp:
                return v
        return float(np.clip(v, -amp, amp))

    return sample

def make_tld_sampler(amp, alpha, cutoff_frac, neg_weight, momentum, rng):
    """
    Truncated Lévy Distribution sampler.

    The TLD has three regions:
      - Central Lévy-stable-like peak (|x| < cutoff)
      - Near-zero gap between cutoff and tail humps
      - Separate tail humps at ±amp (negative slightly heavier)

    Parameters
    ----------
    amp         : float – maximum dither value (%/month), hard clip boundary
    alpha       : float – Lévy stability index (0 < alpha < 2); lower = more
                  peaked center. Typical equity value ~1.5
    cutoff_frac : float – fraction of amp where center distribution is clipped
                  and tail humps begin. Typical value 0.6–0.75
    neg_weight  : float – relative weight of negative tail hump vs positive
                  (>0.5 means negative tail is heavier, matching real equity
                  return asymmetry). Range 0.5–0.8
    momentum    : float – autocorrelation of sign (regime persistence)
    rng         : np.random.Generator
    """
    cutoff = cutoff_frac * amp
    pos_tail_weight = 1.0 - neg_weight

    # Tail hump centers sit just inside the amp boundary
    tail_center_neg = -(cutoff + (amp - cutoff) * 0.6)
    tail_center_pos =  (cutoff + (amp - cutoff) * 0.6)
    tail_sigma      =  (amp - cutoff) * 0.25   # narrow humps

    # Central Lévy-like draw via sum of scaled normals (approximation)
    # A symmetric alpha-stable with index alpha is approximated by
    # mixing Gaussians with Lévy-distributed variances.
    def levy_stable_sample():
        """Approximate symmetric alpha-stable sample, clipped to ±cutoff."""
        v_scaled = 0.0 
        for _ in range(50):
            # Chambers-Mallows-Stuck method for symmetric stable
            u = rng.uniform(-np.pi/2, np.pi/2)
            w = rng.exponential(1.0)
            if abs(alpha - 1.0) < 1e-6:
                v = (2/np.pi) * ((np.pi/2 + u) * np.tan(u)
                                 - np.log((np.pi/2 * w * np.cos(u))
                                          / (np.pi/2 + u)))
            else:
                v = (np.sin(alpha * u) / np.cos(u) ** (1/alpha)
                     * (np.cos(u - alpha * u) / w) ** ((1 - alpha) / alpha))
            # Scale so that typical values sit within ±cutoff
            v_scaled = v * cutoff * 0.3
            if abs(v_scaled) <= cutoff:
                return v_scaled
        return float(np.clip(v_scaled, -cutoff, cutoff))

    # Momentum state: +1 = tending positive, -1 = tending negative, 0 = center
    regime = 0  # start in center

    def sample():
        nonlocal regime

        # Momentum: persist in current regime or randomly switch
        if rng.random() > momentum:
            r = rng.random()
            if r < 0.15:
                regime = -1   # negative tail
            elif r < 0.30:
                regime = 1    # positive tail
            else:
                regime = 0    # center

        if regime == -1:
            # Negative tail hump
            v = tail_center_neg
            for _ in range(30):
                v = rng.normal(tail_center_neg, tail_sigma)
                if -amp <= v <= -cutoff:
                    return v
            return float(np.clip(v, -amp, -cutoff))

        elif regime == 1:
            # Positive tail hump
            v = tail_center_pos
            for _ in range(30):
                v = rng.normal(tail_center_pos, tail_sigma)
                if cutoff <= v <= amp:
                    return v
            return float(np.clip(v, cutoff, amp))

        else:
            # Central Lévy-like region
            return levy_stable_sample()

    return sample

# ══════════════════════════════════════════════════════════════════════════════
# Stress helper
# ══════════════════════════════════════════════════════════════════════════════

def get_stress_params(base_annual_rate, pos_weight, in_stress, sor_severity):
    """
    Returns (effective_annual_rate, effective_pos_weight).
    Each severity step suppresses base rate by 4 ppts and drives
    pos_weight toward 0 (fully negative dither mode).
    """
    if not in_stress:
        return base_annual_rate, pos_weight
    suppression  = sor_severity * 0.04
    weight_shift = sor_severity * (pos_weight / 5.0)
    return base_annual_rate - suppression, max(0.0, pos_weight - weight_shift)

# ══════════════════════════════════════════════════════════════════════════════
# Core simulation
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    principal, base_annual_rate, withdrawal_rate, years,
    amplitude, peak_sep, peak_sigma, bias, pos_weight, momentum,
    n_sims, base_seed, initial_age,
    sor_enabled=False, sor_severity=2, sor_start_yr=1, sor_duration=2,
    dither_model='bimodal',
    tld_alpha=1.5, tld_cutoff=0.65, tld_neg_weight=0.65,
):
    base_monthly_rate = base_annual_rate / 12

    def in_stress_window(yr):
        return (sor_enabled
                and sor_start_yr - 1 <= yr < sor_start_yr - 1 + sor_duration)

    def get_yr_params(yr):
        """Returns (annual_rate, effective_pos_weight) for a given year."""
        return get_stress_params(
            base_annual_rate, pos_weight,
            in_stress_window(yr), sor_severity
        )

    def build_sampler(rng, yr_pos_wt):
        """Constructs the appropriate sampler for the current regime."""
        if dither_model == 'tld':
            return make_tld_sampler(
                amplitude, tld_alpha, tld_cutoff,
                tld_neg_weight, momentum, rng
            )
        else:
            return make_bimodal_sampler(
                amplitude, peak_sep, peak_sigma, bias,
                yr_pos_wt, momentum, rng
            )

    # ── Baseline: no dither, RMD applied, stress applied ─────────────────────
    baseline    = np.empty(years + 1)
    baseline[0] = principal
    bal         = float(principal)

    for yr in range(years):
        age              = initial_age + yr
        yr_rate, _       = get_yr_params(yr)
        monthly_rate     = yr_rate / 12
        rmd_annual       = get_rmd(bal, age)
        voluntary_annual = bal * withdrawal_rate
        monthly_wd       = max(voluntary_annual, rmd_annual) / 12
        for _ in range(12):
            bal = bal * (1 + monthly_rate) - monthly_wd
            bal = max(bal, 0.0)
        baseline[yr + 1] = bal

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    all_balances    = np.empty((n_sims, years + 1))
    all_withdrawals = np.empty((n_sims, years))
    rmd_binding     = np.zeros((n_sims, years), dtype=bool)

    for s in range(n_sims):
        rng            = np.random.default_rng(base_seed + s)
        bal            = float(principal)
        all_balances[s, 0] = bal

        # Build initial sampler before the year loop
        _, init_pos_wt = get_yr_params(0)
        sampler        = build_sampler(rng, init_pos_wt)
        prev_in_stress = in_stress_window(0)

        for yr in range(years):
            age    = initial_age + yr
            stress = in_stress_window(yr)
            yr_rate, yr_pos_wt = get_yr_params(yr)
            yr_monthly_rate    = yr_rate / 12

            # Rebuild sampler only when crossing the stress boundary
            if stress != prev_in_stress:
                sampler = build_sampler(rng, yr_pos_wt)
            prev_in_stress = stress

            rmd_annual         = get_rmd(bal, age)
            voluntary_annual   = bal * withdrawal_rate
            rmd_binding[s, yr] = rmd_annual > voluntary_annual
            monthly_wd         = max(voluntary_annual, rmd_annual) / 12

            year_withdrawn = 0.0
            for _ in range(12):
                if bal <= 0:
                    break
                dither_pct   = sampler()
                monthly_rate = yr_monthly_rate + dither_pct / 100.0
                bal          = bal * (1 + monthly_rate)
                actual_wd    = min(monthly_wd, bal)
                bal         -= actual_wd
                year_withdrawn += actual_wd
                bal          = max(bal, 0.0)

            all_balances[s, yr + 1] = bal
            all_withdrawals[s, yr]  = year_withdrawn

    return baseline, all_balances, all_withdrawals, rmd_binding

# ══════════════════════════════════════════════════════════════════════════════
# Chart drawing functions  (each draws into a supplied axes object)
# ══════════════════════════════════════════════════════════════════════════════

def draw_balance_chart(ax, years_axis, all_balances, baseline,
                       med_bal, p5_bal, p95_bal,
                       p, depleted, first_dep_yr):
    for s in range(p["n_sims"]):
        ax.plot(years_axis, all_balances[s], color=BLUE, alpha=0.10, lw=0.7)

    ax.fill_between(years_axis, p5_bal, p95_bal, color=CORAL, alpha=0.10)
    ax.plot(years_axis, p5_bal,  color=CORAL, lw=1.2, ls='--', alpha=0.7)
    ax.plot(years_axis, p95_bal, color=CORAL, lw=1.2, ls='--', alpha=0.7,
            label='5th / 95th percentile')
    ax.plot(years_axis, med_bal,  color=BLUE,  lw=2.5, label='Median balance')
    ax.plot(years_axis, baseline, color=GREEN, lw=2.0, ls='--',
            label='Baseline (no dither)')

    # Stress shading
    if p.get("sor_enabled"):
        s0 = p["sor_start_yr"] - 1
        s1 = s0 + p["sor_duration"]
        ax.axvspan(s0, min(s1, p["years"]), color=CORAL, alpha=0.08,
                   label='Stress period')
        ax.annotate(
            f'Stress: sev {p["sor_severity"]}, {p["sor_duration"]}yr',
            xy=(s0 + 0.15, ax.get_ylim()[1] * 0.80),
            fontsize=8, color=CORAL,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=CORAL, alpha=0.75),
        )

    # RMD start line
    if p["rmd_start_yr"] < p["years"]:
        ax.axvline(p["rmd_start_yr"], color=AMBER, lw=1.3, ls=':', alpha=0.8)
        ax.annotate(
            f'RMDs begin\n(age {p["initial_age"] + p["rmd_start_yr"]})',
            xy=(p["rmd_start_yr"] + 0.2, ax.get_ylim()[1] * 0.88),
            fontsize=8, color=AMBER,
        )

    ax.set_xlim(0, p["years"])
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Portfolio balance', fontsize=10)
    ax.set_title(
        f'Portfolio balance  |  {p["n_sims"]} simulations  |  '
        f'base rate {p["base_annual_rate"]*100:.1f}%  |  '
        f'withdrawal {p["withdrawal_rate"]*100:.1f}% of balance  |  '
        f'initial age {p["initial_age"]}',
        fontsize=10, fontweight='bold'
    )
    ax.yaxis.set_major_formatter(dollar_formatter)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', color=LGRAY, alpha=0.35, lw=0.5)

    if depleted > 0:
        msg = (f'{depleted}/{p["n_sims"]} portfolios depleted'
               + (f' — earliest yr {first_dep_yr}' if first_dep_yr else ''))
        ax.annotate(msg, xy=(0.02, 0.05), xycoords='axes fraction',
                    fontsize=8, color=CORAL,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec=CORAL, alpha=0.85))


def draw_withdrawal_chart(ax, year_labels, all_withdrawals,
                          med_wd, p5_wd, p95_wd,
                          p, rmd_first_yr):
    for s in range(p["n_sims"]):
        ax.plot(year_labels, all_withdrawals[s], color=PURPLE, alpha=0.10, lw=0.7)

    ax.fill_between(year_labels, p5_wd, p95_wd, color=PURPLE, alpha=0.10)
    ax.plot(year_labels, p5_wd,  color=PURPLE, lw=1.2, ls='--', alpha=0.7)
    ax.plot(year_labels, p95_wd, color=PURPLE, lw=1.2, ls='--', alpha=0.7,
            label='5th / 95th percentile')
    ax.plot(year_labels, med_wd, color=PURPLE, lw=2.5, label='Median withdrawal')

    # Stress shading (lagged one year)
    if p.get("sor_enabled"):
        s0 = p["sor_start_yr"] - 1
        s1 = s0 + p["sor_duration"]
        ax.axvspan(s0 + 1, min(s1 + 1, p["years"]),
                   color=CORAL, alpha=0.08, label='Stress impact (lagged 1yr)')
        ax.annotate(
            f'Stress impact\n(lagged 1yr)',
            xy=(s0 + 1.2, ax.get_ylim()[1] * 0.80),
            fontsize=8, color=CORAL,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=CORAL, alpha=0.75),
        )

    # RMD binding line
    if rmd_first_yr is not None and 0 < rmd_first_yr < p["years"]:
        ax.axvline(rmd_first_yr + 1, color=AMBER, lw=1.3, ls=':', alpha=0.8,
                   label=f'RMD binding (yr {rmd_first_yr + 1})')
        ax.annotate(
            f'RMD binding\n(yr {rmd_first_yr + 1})',
            xy=(rmd_first_yr + 1.2, ax.get_ylim()[1] * 0.88),
            fontsize=8, color=AMBER,
        )

    ax.set_xlim(1, p["years"])
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Annual withdrawal', fontsize=10)
    ax.set_title(
        'Actual annual withdrawals  '
        '(higher of voluntary rate vs RMD — drops to zero when depleted)',
        fontsize=10, fontweight='bold'
    )
    ax.yaxis.set_major_formatter(dollar_formatter)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', color=LGRAY, alpha=0.35, lw=0.5)


def draw_dither_pdf(ax, amplitude, peak_sep, peak_sigma, bias, pos_weight,
                    dither_model='bimodal',
                    tld_alpha=1.5, tld_cutoff=0.65, tld_neg_weight=0.65):
    x = np.linspace(-amplitude * 1.1, amplitude * 1.1, 500)

    if dither_model == 'bimodal':
        mu_pos =  peak_sep * amplitude + bias * amplitude
        mu_neg = -peak_sep * amplitude + bias * amplitude
        sigma  =  peak_sigma * amplitude

        pdf_pos = pos_weight       * sp_norm.pdf(x, mu_pos, sigma)
        pdf_neg = (1 - pos_weight) * sp_norm.pdf(x, mu_neg, sigma)
        pdf_tot = pdf_pos + pdf_neg

        ax.fill_between(x, pdf_tot, alpha=0.22, color=PURPLE)
        ax.plot(x, pdf_tot, color=PURPLE, lw=2.0, label='Total PDF')
        ax.plot(x, pdf_pos, color=BLUE,   lw=1.3, ls='--', alpha=0.8,
                label=f'Pos mode (w={pos_weight:.2f}, μ={mu_pos:+.3f}%)')
        ax.plot(x, pdf_neg, color=CORAL,  lw=1.3, ls='--', alpha=0.8,
                label=f'Neg mode (w={1-pos_weight:.2f}, μ={mu_neg:+.3f}%)')
        ax.axvline(0,                color=LGRAY,   lw=0.8, ls=':')
        ax.axvline(bias * amplitude, color='black', lw=1.2, ls='-', alpha=0.6,
                   label=f'Bias ({bias*amplitude:+.3f}%)')

    else:  # TLD
        pdf_tot, pdf_ctr, pdf_neg_tail, pdf_pos_tail = tld_pdf_components(
            x, amplitude, tld_alpha, tld_cutoff, tld_neg_weight
        )
        cutoff = tld_cutoff * amplitude
        ax.fill_between(x, pdf_tot, alpha=0.22, color=PURPLE)
        ax.plot(x, pdf_tot,      color=PURPLE, lw=2.0, label='Total PDF')
        ax.plot(x, pdf_ctr,      color=BLUE,   lw=1.3, ls='--', alpha=0.8,
                label=f'Center (α={tld_alpha:.2f})')
        ax.plot(x, pdf_neg_tail, color=CORAL,  lw=1.3, ls='--', alpha=0.8,
                label=f'Neg tail (w={tld_neg_weight:.2f})')
        ax.plot(x, pdf_pos_tail, color=GREEN,  lw=1.3, ls='--', alpha=0.8,
                label=f'Pos tail (w={1-tld_neg_weight:.2f})')
        ax.axvline( cutoff, color=LGRAY, lw=0.8, ls=':', alpha=0.7)
        ax.axvline(-cutoff, color=LGRAY, lw=0.8, ls=':', alpha=0.7,
                   label=f'Cutoff (±{cutoff:.3f}%)')
        
    ax.set_xlim(-amplitude * 1.1, amplitude * 1.1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Monthly rate dither (%/month)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Bimodal dither distribution', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', color=LGRAY, alpha=0.4, lw=0.5)

def tld_pdf_components(x, amp, alpha, cutoff_frac, neg_weight):
    """Returns total PDF array for TLD for display purposes."""
    cutoff = cutoff_frac * amp
    pos_tail_weight = 1.0 - neg_weight

    tail_center_neg = -(cutoff + (amp - cutoff) * 0.6)
    tail_center_pos =  (cutoff + (amp - cutoff) * 0.6)
    tail_sigma      =  (amp - cutoff) * 0.25

    # Central: approximate Lévy stable PDF via scipy
    from scipy.stats import levy_stable
    # Scale so central mass fits within ±cutoff
    scale = cutoff * 0.3 / 2.0
    pdf_center = levy_stable.pdf(x, alpha=alpha, beta=0, scale=scale, loc=0)
    pdf_center = np.asarray(pdf_center, dtype=float)
    # Zero out outside ±cutoff (the truncation)
    pdf_center = np.where(np.abs(x) <= cutoff, pdf_center, 0.0)

    # Tail humps
    pdf_neg_tail = neg_weight       * sp_norm.pdf(x, tail_center_neg, tail_sigma)
    pdf_pos_tail = pos_tail_weight  * sp_norm.pdf(x, tail_center_pos, tail_sigma)
    # Zero out tails inside cutoff
    pdf_neg_tail = np.where(x <= -cutoff, pdf_neg_tail, 0.0)
    pdf_pos_tail = np.where(x >=  cutoff, pdf_pos_tail, 0.0)

    # Normalize each component so the total is a proper display PDF
    dx = x[1] - x[0]
    center_mass   = pdf_center.sum()   * dx
    neg_tail_mass = pdf_neg_tail.sum() * dx
    pos_tail_mass = pdf_pos_tail.sum() * dx
    total_mass    = center_mass + neg_tail_mass + pos_tail_mass

    center_frac   = 0.70   # 70% of probability in center, 30% in tails
    tail_frac     = 1.0 - center_frac

    if center_mass > 0:
        pdf_center   = pdf_center   * (center_frac  / center_mass  * dx)
    if neg_tail_mass > 0:
        pdf_neg_tail = pdf_neg_tail * (tail_frac * neg_weight      / neg_tail_mass * dx)
    if pos_tail_mass > 0:
        pdf_pos_tail = pdf_pos_tail * (tail_frac * pos_tail_weight / pos_tail_mass * dx)

    pdf_total = pdf_center + pdf_neg_tail + pdf_pos_tail
    return pdf_total, pdf_center, pdf_neg_tail, pdf_pos_tail

def draw_param_table(ax, p, med_bal, p5_bal, p95_bal, depleted, first_dep_yr):
    ax.axis('off')
    rows = [
        ('── Portfolio ──',        ''),
        ('Initial principal',      fmt_dollars(p['principal'])),
        ('Base annual rate',       f'{p["base_annual_rate"]*100:.1f}%'),
        ('Withdrawal rate',        f'{p["withdrawal_rate"]*100:.1f}% of balance'),
        ('Initial age',            str(p['initial_age'])),
        ('Years',                  str(p['years'])),
        ('Simulations',            str(p['n_sims'])),
        ('── Dither ──',           ''),
        ('Amplitude',              f'±{p["amplitude"]:.2f} %/month'),
        ('Peak separation',        f'{p["peak_sep"]:.2f} × amp'),
        ('Peak width σ',           f'{p["peak_sigma"]:.3f} × amp'),
        ('Bias',                   f'{p["bias"]:+.3f} × amp'),
        ('Pos mode weight',        f'{p["pos_weight"]:.2f}'),
        ('Momentum',               f'{p["momentum"]:.2f}'),
        ('── Stress ──',           ''),
        ('Enabled',                str(p.get("sor_enabled", False))),
        ('Severity',               str(p.get("sor_severity", "—"))),
        ('Start year',             str(p.get("sor_start_yr", "—"))),
        ('Duration',               str(p.get("sor_duration", "—"))),
        ('── Results ──',          ''),
        ('Median final balance',   fmt_dollars(med_bal[-1])),
        ('5th pct final bal.',     fmt_dollars(p5_bal[-1])),
        ('95th pct final bal.',    fmt_dollars(p95_bal[-1])),
        ('Portfolios depleted',    f'{depleted} / {p["n_sims"]}'),
        ('Earliest depletion',     f'Year {first_dep_yr}' if first_dep_yr else 'None'),
    ]
    y0 = 1.0
    dy = 1.0 / (len(rows) + 1)
    for label, value in rows:
        bold  = '──' in label
        color = '#444441' if bold else 'black'
        ax.text(0.02, y0, label, transform=ax.transAxes,
                fontsize=8, va='top', color=color,
                fontweight='bold' if bold else 'normal')
        ax.text(0.98, y0, value, transform=ax.transAxes,
                fontsize=8, va='top', ha='right', color='#185FA5')
        y0 -= dy
    ax.set_title('Parameters & Results', fontsize=10, fontweight='bold')


def build_export_figure(years_axis, year_labels,
                        all_balances, baseline, all_withdrawals,
                        med_bal, p5_bal, p95_bal,
                        med_wd,  p5_wd,  p95_wd,
                        p, depleted, first_dep_yr, rmd_first_yr):
    """Assembles all four panels into one figure for the download PNG."""
    fig = plt.figure(figsize=(14, 18))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[2.2, 2.2, 1.5],
        hspace=0.42, wspace=0.32
    )
    ax_bal  = fig.add_subplot(gs[0, :])
    ax_wd   = fig.add_subplot(gs[1, :])
    ax_pdf  = fig.add_subplot(gs[2, 0])
    ax_info = fig.add_subplot(gs[2, 1])

    draw_balance_chart(ax_bal, years_axis, all_balances, baseline,
                       med_bal, p5_bal, p95_bal, p, depleted, first_dep_yr)
    draw_withdrawal_chart(ax_wd, year_labels, all_withdrawals,
                          med_wd, p5_wd, p95_wd, p, rmd_first_yr)
    draw_dither_pdf(ax_pdf,
                p["amplitude"], p["peak_sep"], p["peak_sigma"],
                p["bias"], p["pos_weight"],
                dither_model=p.get("dither_model") or  "bimodal",
                tld_alpha=p.get("tld_alpha", 1.5),
                tld_cutoff=p.get("tld_cutoff", 0.65),
                tld_neg_weight=p.get("tld_neg_weight", 0.65))
    draw_param_table(ax_info, p, med_bal, p5_bal, p95_bal,
                     depleted, first_dep_yr)

    fig.suptitle('Retirement Portfolio Monte Carlo Simulation',
                 fontsize=13, fontweight='bold')
    fig.subplots_adjust(top=0.9)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# Streamlit page
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Retirement Portfolio Simulator",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Retirement Portfolio Simulator")
st.caption(
    "Monte Carlo simulation with bimodal monthly rate dither, "
    "dynamic withdrawals, IRS Required Minimum Distributions, "
    "and optional sequence-of-returns stress testing."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    st.header("Portfolio")
    principal = st.number_input(
        "Initial deposit ($)",
        min_value=100_000, max_value=20_000_000,
        value=2_000_000, step=100_000, format="%d",
    )
    base_rate_pct = st.slider(
        "Base annual return rate (%)",
        min_value=1.0, max_value=20.0, value=5.0, step=0.5,
    )
    withdrawal_pct = st.slider(
        "Annual withdrawal rate (% of current balance)",
        min_value=3.0, max_value=10.0, value=4.0, step=0.1,
    )
    initial_age = st.slider(
        "Age at first withdrawal",
        min_value=50, max_value=80, value=67, step=1,
    )
    years = st.slider(
        "Simulation horizon (years)",
        min_value=5, max_value=30, value=25, step=1,
    )
    n_sims = st.slider(
        "Number of simulations",
        min_value=10, max_value=200, value=50, step=10,
    )
    random_seed = st.number_input(
        "Random seed",
        min_value=0, max_value=9999, value=42, step=1,
    )

    st.divider()
    st.header("Monthly Rate Dither")
    st.caption("Noise added to the monthly return rate.")

    dither_model = st.radio(
        "Dither model",
        options=["bimodal", "tld"],
        format_func=lambda x: "Bimodal Gaussian" if x == "bimodal" else "Truncated Lévy (TLD)",
        horizontal=True,
    ) or "bimodal"

    amplitude = st.slider(
        "Max amplitude ±(%/month)",
        min_value=0.1, max_value=1.5, value=0.75, step=0.05,
    )
    momentum = st.slider(
        "Momentum (mode autocorrelation)",
        min_value=0.00, max_value=0.95, value=0.65, step=0.05,
        help="Probability of staying in the current regime each month.",
    )

    # Bimodal-specific controls
    bimodal_disabled = (dither_model == "tld")
    st.caption("**Bimodal Gaussian parameters**" if not bimodal_disabled
               else "*Bimodal parameters (inactive)*")
    peak_sep = st.slider(
        "Peak separation (fraction of amplitude)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        disabled=bimodal_disabled,
        help="Moves the two Gaussian peaks symmetrically away from zero.",
    )
    peak_sigma = st.slider(
        "Peak width σ (fraction of amplitude)",
        min_value=0.05, max_value=0.60, value=0.25, step=0.025,
        disabled=bimodal_disabled,
    )
    bias = st.slider(
        "Bias — median shift (fraction of amplitude)",
        min_value=-0.50, max_value=0.50, value=0.00, step=0.025,
        disabled=bimodal_disabled,
    )
    pos_weight = st.slider(
        "Positive mode weight",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        disabled=bimodal_disabled,
    )

    # TLD-specific controls
    tld_disabled = (dither_model == "bimodal")
    st.caption("**Truncated Lévy parameters**" if not tld_disabled
               else "*TLD parameters (inactive)*")
    tld_alpha = st.slider(
        "Lévy stability index α",
        min_value=0.5, max_value=1.99, value=1.5, step=0.05,
        disabled=tld_disabled,
        help="Lower = more peaked center and heavier tails. Typical equity ~1.5",
    )
    tld_cutoff = st.slider(
        "Cutoff (fraction of amplitude)",
        min_value=0.30, max_value=0.85, value=0.65, step=0.05,
        disabled=tld_disabled,
        help="Where the central distribution ends and the gap begins.",
    )
    tld_neg_weight = st.slider(
        "Negative tail weight",
        min_value=0.50, max_value=0.90, value=0.65, step=0.05,
        disabled=tld_disabled,
        help="Relative weight of negative vs positive tail hump. >0.5 = heavier negative tail.",
    )

    st.divider()
    st.header("Sequence of Returns Stress")
    sor_enabled = st.toggle("Enable stress period", value=False)
    sor_severity = st.slider(
        "Severity (0 = mild, 5 = severe)",
        min_value=0, max_value=5, value=2, step=1,
        disabled=not sor_enabled,
        help=(
            "Each step suppresses the base annual return by ~4 ppts "
            "and shifts dither toward the negative mode. "
            "Severity 5 on a 7% base → sustained −13% annual return."
        ),
    )
    sor_start_yr = st.slider(
        "Stress starts in year",
        min_value=1, max_value=max(1, years - 1), value=1, step=1,
        disabled=not sor_enabled,
        help="Year 1 = immediately at retirement (worst case).",
    )
    sor_duration = st.slider(
        "Duration (years)",
        min_value=1, max_value=5, value=2, step=1,
        disabled=not sor_enabled,
    )

    st.divider()
    run = st.button("▶  Run Simulation", type="primary", use_container_width=True)

# ── Derived scalars ───────────────────────────────────────────────────────────

base_annual_rate = base_rate_pct / 100.0
withdrawal_rate  = withdrawal_pct / 100.0
rmd_start_yr     = max(0, 73 - initial_age)

# ══════════════════════════════════════════════════════════════════════════════
# Dither PDF preview  (always live — no run needed)
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Dither Distribution Preview")
st.caption("Updates live as you adjust dither controls. Click ▶ Run Simulation to update portfolio charts.")

fig_pdf, ax_pdf_prev = plt.subplots(figsize=(7, 2.6))
draw_dither_pdf(ax_pdf_prev, amplitude, peak_sep, peak_sigma, bias, pos_weight,
                dither_model=dither_model or "bimodal",
                tld_alpha=tld_alpha,
                tld_cutoff=tld_cutoff,
                tld_neg_weight=tld_neg_weight)
fig_pdf.tight_layout()
st.pyplot(fig_pdf, use_container_width=True)
plt.close(fig_pdf)

# ══════════════════════════════════════════════════════════════════════════════
# Run simulation on button press
# ══════════════════════════════════════════════════════════════════════════════

if run:
    with st.spinner("Running simulations…"):
        results = run_simulation(
            principal        = principal,
            base_annual_rate = base_annual_rate,
            withdrawal_rate  = withdrawal_rate,
            years            = years,
            amplitude        = amplitude,
            peak_sep         = peak_sep,
            peak_sigma       = peak_sigma,
            bias             = bias,
            pos_weight       = pos_weight,
            momentum         = momentum,
            n_sims           = n_sims,
            base_seed        = int(random_seed),
            initial_age      = initial_age,
            sor_enabled      = sor_enabled,
            sor_severity     = sor_severity,
            sor_start_yr     = sor_start_yr,
            sor_duration     = sor_duration,
            dither_model     = dither_model,
            tld_alpha        = tld_alpha,
            tld_cutoff       = tld_cutoff,
            tld_neg_weight   = tld_neg_weight,
        )
    st.session_state["results"] = results
    st.session_state["params"]  = dict(
        principal        = principal,
        base_annual_rate = base_annual_rate,
        withdrawal_rate  = withdrawal_rate,
        years            = years,
        amplitude        = amplitude,
        peak_sep         = peak_sep,
        peak_sigma       = peak_sigma,
        bias             = bias,
        pos_weight       = pos_weight,
        momentum         = momentum,
        n_sims           = n_sims,
        initial_age      = initial_age,
        rmd_start_yr     = rmd_start_yr,
        sor_enabled      = sor_enabled,
        sor_severity     = sor_severity,
        sor_start_yr     = sor_start_yr,
        sor_duration     = sor_duration,
        dither_model     = dither_model,
        tld_alpha        = tld_alpha,
        tld_cutoff       = tld_cutoff,
        tld_neg_weight   = tld_neg_weight,
    )

if "results" not in st.session_state:
    st.info("👈  Adjust parameters in the sidebar, then click **▶ Run Simulation**.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Unpack results and compute stats
# ══════════════════════════════════════════════════════════════════════════════

baseline, all_balances, all_withdrawals, rmd_binding = st.session_state["results"]
p = st.session_state["params"]

years_axis  = np.arange(p["years"] + 1)
year_labels = np.arange(1, p["years"] + 1)

med_bal = np.median(all_balances,    axis=0)
p5_bal  = pct(all_balances,   5,    axis=0)
p95_bal = pct(all_balances,  95,    axis=0)
med_wd  = np.median(all_withdrawals, axis=0)
p5_wd   = pct(all_withdrawals,  5,  axis=0)
p95_wd  = pct(all_withdrawals, 95,  axis=0)

# Depletion stats
depleted     = int(np.any(all_balances[:, 1:] == 0, axis=1).sum())
first_dep_yr = None
mask = np.any(all_balances[:, 1:] == 0, axis=1)
if mask.any():
    zero_years   = np.argmax(all_balances[:, 1:] == 0, axis=1)
    first_dep_yr = int(zero_years[mask].min()) + 1

# RMD binding stats
rmd_first_yr = None
rmd_cols = np.any(rmd_binding, axis=0)
if rmd_cols.any():
    rmd_first_yr = int(np.argmax(rmd_cols))

# ══════════════════════════════════════════════════════════════════════════════
# Summary metrics
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Simulation Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Median final balance",   fmt_dollars(med_bal[-1]))
c2.metric("5th pct final balance",  fmt_dollars(p5_bal[-1]))
c3.metric("95th pct final balance", fmt_dollars(p95_bal[-1]))
c4.metric("Portfolios depleted",    f"{depleted} / {p['n_sims']}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Median withdrawal yr 1",     fmt_dollars(med_wd[0]))
c6.metric("Median withdrawal final yr", fmt_dollars(med_wd[-1]))
rmd_age_label = (f"Age {p['initial_age'] + p['rmd_start_yr']}"
                 if p["rmd_start_yr"] < p["years"] else "Beyond horizon")
c7.metric("RMD kicks in",       rmd_age_label)
c8.metric("Earliest depletion", f"Year {first_dep_yr}" if first_dep_yr else "None")

# ══════════════════════════════════════════════════════════════════════════════
# Screen charts  (balance + withdrawals only)
# ══════════════════════════════════════════════════════════════════════════════

fig_screen, (ax_s_bal, ax_s_wd) = plt.subplots(
    2, 1, figsize=(13, 10), gridspec_kw={'hspace': 0.40}
)
draw_balance_chart(ax_s_bal, years_axis, all_balances, baseline,
                   med_bal, p5_bal, p95_bal, p, depleted, first_dep_yr)
draw_withdrawal_chart(ax_s_wd, year_labels, all_withdrawals,
                      med_wd, p5_wd, p95_wd, p, rmd_first_yr)
#fig_screen.suptitle('Retirement Portfolio Monte Carlo Simulation', fontsize=13, fontweight='bold')
fig_screen.subplots_adjust(top=0.95)
st.pyplot(fig_screen, use_container_width=True)
plt.close(fig_screen)

# ══════════════════════════════════════════════════════════════════════════════
# Export figure  (all four panels) → download button
# ══════════════════════════════════════════════════════════════════════════════

fig_export = build_export_figure(
    years_axis, year_labels,
    all_balances, baseline, all_withdrawals,
    med_bal, p5_bal, p95_bal,
    med_wd,  p5_wd,  p95_wd,
    p, depleted, first_dep_yr, rmd_first_yr,
)
buf = io.BytesIO()
fig_export.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
plt.close(fig_export)

st.download_button(
    label="⬇  Download full report (charts + dither PDF + parameters)",
    data=buf,
    file_name="retirement_sim_report.png",
    mime="image/png",
)
