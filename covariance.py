"""
extract_lyc_indicators.py
=========================
Extracts reliable indirect LyC escape fraction indicators from the SPHINX20
simulation catalogue, then computes three correlation matrices:

  1. Pearson       – linear relationships
  2. Spearman      – monotonic (nonlinear) relationships
  3. Mutual Info   – any statistical dependence (linear or not)

Directional quantities (dir_0 … dir_9) are treated as independent galaxy
sightlines: every (halo, direction) pair becomes its own row.

Indicators
----------
  1.  O32          = log10( [OIII]5007 / ([OII]3726+[OII]3729) )
  2.  R23          = log10( ([OIII]4959+[OIII]5007 + [OII]3726+[OII]3729) / Hβ )
  3.  Ne3O2        = log10( [NeIII]3869 / ([OII]3726+[OII]3729) )
  4.  O3Hb         = log10( [OIII]5007 / Hβ )
  5.  N2Ha         = log10( [NII]6584 / Hα )
  6.  S2_deficit   = log10( ([SII]6716+[SII]6731) / Hα )
  7.  Balmer_dec   = Hα / Hβ
  8.  EW_Hb        = EW(Hβ)   [line / continuum]
  9.  EW_O3        = EW([OIII]5007)
 10.  beta_UV      = UV continuum slope β
 11.  xi_ion       = log10(ξ_ion)   [intrinsic only – same for all 10 directions]
 12.  MAB_1500     = absolute UV magnitude at 1500 Å
 13.  E_BV         = E(B-V) nebular attenuation
 14.  fesc_dir     = directional LyC escape fraction (target)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

SIM_CATALOGUE = '/home/mezziati/Documents/IAP/SPHINX20/data/all_basic_data.csv'
# OBS_CATALOGUE = '/home/mezziati/Documents/IAP/SPHINX20/data/flury.csv'
OUTPUT_DIR    = '/home/mezziati/Documents/IAP/SPHINX20/sphinx_analysis/outputs/'

N_DIRS   = 10
LOG_CLIP = 1e-30

# indicators only (no fesc)
INDICATOR_NAMES = [
    "O32", "R23", "Ne3O2", "O3Hb", "N2Ha",
    "S2_deficit", "Balmer_dec", "EW_Hb", "EW_O3",
    "beta_UV", "xi_ion", "MAB_1500", "E_BV", "redshift"
]

# all columns going into the full correlation matrices (indicators + target)
ALL_COLS = INDICATOR_NAMES + ["fesc_dir"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_log10(arr, clip=LOG_CLIP):
    a = np.where(np.isfinite(arr) & (arr > 0), arr, clip)
    return np.log10(a)


def col(df, name, default=np.nan):
    if name in df.columns:
        return df[name].values.astype(float)
    return np.full(len(df), default)


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION INDICATOR EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_sim_directional(df, n_dirs=N_DIRS):
    """
    Expand the simulation table: one row per (halo, direction).
    Each (halo, dir) pair is treated as an independent galaxy.
    """
    rows = []

    for d in range(n_dirs):
        suf = f"_dir_{d}"

        # ── directional line fluxes ───────────────────────────────────────
        Ha       = col(df, f"HI_6562.8{suf}",        col(df, "HI_6562.8_int"))
        Hb       = col(df, f"HI_4861.32{suf}",       col(df, "HI_4861.32_int"))
        OII      = col(df, f"OII_3726.03{suf}",      col(df, "OII_3726.03_int")) + \
                   col(df, f"OII_3728.81{suf}",      col(df, "OII_3728.81_int"))
        OIII5007 = col(df, f"OIII_5006.84{suf}",     col(df, "OIII_5006.84_int"))
        OIII4959 = col(df, f"OIII_4958.91{suf}",     col(df, "OIII_4958.91_int"))
        NeIII    = col(df, f"NeIII_3868.76{suf}",    col(df, "NeIII_3868.76_int"))
        NII      = col(df, f"NII_6583.45{suf}",      col(df, "NII_6583.45_int"))

        # [SII] only available as intrinsic in the sim
        SII      = col(df, "S__2_6716.44A_int") + col(df, "S__2_6730.82A_int")

        # ── directional continuum → EW ────────────────────────────────────
        cont_Hb  = col(df, f"cont_4861{suf}",        col(df, "cont_4861_int"))
        cont_O3  = col(df, f"cont_5008{suf}",        col(df, "cont_5008_int"))
        EW_Hb    = np.where(cont_Hb > 0, Hb / cont_Hb, np.nan)
        EW_O3    = np.where(cont_O3 > 0, OIII5007 / cont_O3, np.nan)

        # ── other directional quantities ──────────────────────────────────
        beta     = col(df, f"beta_dir_{d}_sn",       col(df, "beta_int_sn"))
        mab      = col(df, f"MAB_1500{suf}",         col(df, "MAB_1500_int"))
        ebmv     = col(df, f"ebmv{suf}")
        fesc     = col(df, f"fesc_dir_{d}")

        # ── xi_ion: intrinsic only (same for all 10 directions of a halo) ─
        xi       = col(df, "xi_ion")
        redshift = col(df, "redshift")

        chunk = pd.DataFrame({
            "halo_id"    : col(df, "halo_id").astype(int),
            "dir"        : d,
            "fesc_dir"   : fesc,
            "O32"        : safe_log10(OIII5007)                        - safe_log10(OII),
            "R23"        : safe_log10(OIII5007 + OIII4959 + OII)      - safe_log10(Hb),
            "Ne3O2"      : safe_log10(NeIII)                           - safe_log10(OII),
            "O3Hb"       : safe_log10(OIII5007)                        - safe_log10(Hb),
            "N2Ha"       : safe_log10(NII)                             - safe_log10(Ha),
            "S2_deficit" : safe_log10(SII)                             - safe_log10(Ha),
            "Balmer_dec" : np.where(Hb > 0, Ha / Hb, np.nan),
            "EW_Hb"      : EW_Hb,
            "EW_O3"      : EW_O3,
            "beta_UV"    : beta,
            "xi_ion"     : safe_log10(xi),
            "MAB_1500"   : mab,
            "E_BV"       : ebmv,
            "redshift"   : redshift,
        })
        rows.append(chunk)

    result = pd.concat(rows, ignore_index=True)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def compute_pearson(df, cols):
    """Standard Pearson correlation matrix. Listwise NaN deletion."""
    data = df[cols].dropna()
    corr = pd.DataFrame(np.corrcoef(data.values.T), index=cols, columns=cols)
    return corr, len(data)


def compute_spearman(df, cols):
    """
    Spearman rank correlation matrix.
    Captures any monotonic relationship, not just linear.
    """
    data = df[cols].dropna()
    corr_arr, _ = stats.spearmanr(data.values)
    if np.ndim(corr_arr) == 0:
        corr_arr = np.array([[1.0, float(corr_arr)],
                             [float(corr_arr), 1.0]])
    corr = pd.DataFrame(corr_arr, index=cols, columns=cols)
    return corr, len(data)


def compute_mutual_info(df, cols, target="fesc_dir", n_neighbors=5, random_state=42):
    """
    Mutual information between each indicator and the target (fesc_dir),
    plus a full symmetric MI matrix for all pairs.

    MI values are in nats (non-negative; higher = more dependence).
    Diagonal is set to NaN (MI of a variable with itself = its entropy).
    """
    data = df[cols].dropna()
    indicators = [c for c in cols if c != target]
    y = data[target].values

    # MI of each indicator vs fesc (fast, single call)
    X = data[indicators].values
    mi_vs_target = mutual_info_regression(X, y, n_neighbors=n_neighbors,
                                           random_state=random_state)
    mi_series = pd.Series(mi_vs_target, index=indicators, name="MI_vs_fesc")

    # Full symmetric MI matrix (all pairs)
    n = len(cols)
    mi_matrix = np.full((n, n), np.nan)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                continue
            elif j < i:
                mi_matrix[i, j] = mi_matrix[j, i]
            else:
                mi_val = mutual_info_regression(
                    data[[c1]].values, data[c2].values,
                    n_neighbors=n_neighbors, random_state=random_state
                )[0]
                mi_matrix[i, j] = mi_val

    mi_df = pd.DataFrame(mi_matrix, index=cols, columns=cols)
    return mi_df, mi_series, len(data)


# ══════════════════════════════════════════════════════════════════════════════
# RANKED SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_ranking_table(pearson_df, spearman_df, mi_series, target="fesc_dir"):
    """
    Ranked table: Pearson r, Spearman ρ, MI for each indicator vs fesc_dir.
    Sorted by |Spearman ρ| descending.
    """
    indicators = [c for c in pearson_df.columns if c != target]
    table = pd.DataFrame({
        "Pearson_r"   : pearson_df.loc[indicators, target],
        "Spearman_rho": spearman_df.loc[indicators, target],
        "MI_vs_fesc"  : mi_series.reindex(indicators),
    })
    table = table.reindex(table["Spearman_rho"].abs().sort_values(ascending=False).index)
    return table


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(matrix_df, title, out_path, cmap="RdBu_r",
                 vmin=None, vmax=None, fmt=".2f", annot_thresh=0.6):
    if not HAS_MPL:
        print("[WARN] matplotlib not available – skipping plot.")
        return
    n = len(matrix_df)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.75), max(5, n * 0.7)))
    im = ax.imshow(matrix_df.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(matrix_df.index, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = matrix_df.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > annot_thresh * abs(vmax or 1) else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    fontsize=7, color=color)
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_ranking_bar(ranking_df, out_path):
    """Horizontal bar chart comparing Pearson, Spearman, MI for each indicator."""
    if not HAS_MPL:
        return
    indicators = ranking_df.index.tolist()
    n = len(indicators)
    x = np.arange(n)
    width = 0.25

    # normalise MI to [0, 1] so it sits on the same axis as correlations
    mi_raw  = ranking_df["MI_vs_fesc"].values
    mi_norm = mi_raw / (mi_raw.max() + 1e-30)

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5)))
    ax.barh(x - width, ranking_df["Pearson_r"].values,    width, label="Pearson r",   color="#4C72B0")
    ax.barh(x,         ranking_df["Spearman_rho"].values, width, label="Spearman ρ",  color="#DD8452")
    ax.barh(x + width, mi_norm,                           width, label="MI (norm.)",  color="#55A868")
    ax.set_yticks(x)
    ax.set_yticklabels(indicators, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline( 0.3, color="red",  linewidth=1.2, linestyle="--", label="+0.3 threshold")
    ax.axvline(-0.3, color="blue", linewidth=1.2, linestyle="--", label="−0.3 threshold")
    ax.set_xlabel("Correlation with fesc_dir  (MI normalised to [0, 1])", fontsize=9)
    ax.set_title("Indicator ranking vs fesc_dir", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTS
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df, cols, n_used):
    print(f"\n{'='*60}")
    print(f"  Total rows: {len(df):,}   |   Rows used (no NaN): {n_used:,}")
    print(f"{'='*60}")
    print(f"  {'Indicator':<14}  {'mean':>9}  {'std':>9}  {'min':>9}  {'max':>9}  {'n_valid':>8}")
    print(f"  {'-'*62}")
    for c in cols:
        s = df[c].dropna()
        print(f"  {c:<14}  {s.mean():>9.3f}  {s.std():>9.3f}  "
              f"{s.min():>9.3f}  {s.max():>9.3f}  {len(s):>8,}")


def print_ranking(table):
    print(f"\n{'='*60}")
    print(f"  INDICATOR RANKING vs fesc_dir")
    print(f"  sorted by |Spearman ρ|")
    print(f"{'='*60}")
    print(f"  {'Indicator':<14}  {'Pearson r':>10}  {'Spearman ρ':>11}  {'MI (nats)':>10}")
    print(f"  {'-'*52}")
    for ind, row in table.iterrows():
        print(f"  {ind:<14}  {row['Pearson_r']:>10.3f}  "
              f"{row['Spearman_rho']:>11.3f}  {row['MI_vs_fesc']:>10.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# TOP-INDICATOR SELECTION: MI filter + redundancy removal
# ══════════════════════════════════════════════════════════════════════════════

def select_final_indicators(mi_vs_fesc, spearman_full_df,
                             mi_threshold=0.3, redundancy_threshold=0.8,
                             target="fesc_dir"):
    """
    Two-step indicator selection:

    Step 1 – Keep indicators with MI_norm vs fesc >= mi_threshold
             (MI normalised by max MI vs fesc, so 0.3 = 30% as informative
             as the best predictor — the only meaningful MI normalisation).

    Step 2 – Among the survivors, greedily remove redundancies:
             scan all pairs by |Spearman rho|; when a pair exceeds
             redundancy_threshold, drop whichever has the lower MI vs fesc.
             Repeat until no redundant pair remains.

    Returns
    -------
    kept       : list of final indicator names
    dropped    : list of (dropped, kept_partner, rho, reason) tuples
    mi_norm    : Series of normalised MI values for all candidates
    """
    # Step 1: MI filter
    mi_norm = mi_vs_fesc / (mi_vs_fesc.max() + 1e-30)
    candidates = mi_norm[mi_norm >= mi_threshold].index.tolist()

    if not candidates:
        print(f"  [WARN] No indicators passed MI_norm >= {mi_threshold}.")
        return [], [], mi_norm

    print(f"\n  Step 1 – Indicators with MI_norm >= {mi_threshold} vs fesc ({len(candidates)}):")
    for ind in candidates:
        print(f"    {ind:<14}  MI_norm={mi_norm[ind]:.3f}  MI_raw={mi_vs_fesc[ind]:.4f}")

    # Step 2: greedy redundancy removal
    kept    = list(candidates)
    dropped = []          # (dropped_ind, partner, rho, mi_norm_dropped, mi_norm_kept)

    changed = True
    while changed:
        changed = False
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                a, b = kept[i], kept[j]
                rho = abs(spearman_full_df.loc[a, b])
                if rho >= redundancy_threshold:
                    # drop the one with lower MI vs fesc
                    if mi_vs_fesc[a] >= mi_vs_fesc[b]:
                        loser, winner = b, a
                    else:
                        loser, winner = a, b
                    dropped.append((loser, winner, rho,
                                    mi_norm[loser], mi_norm[winner]))
                    kept.remove(loser)
                    changed = True
                    break           # restart scan after any removal
            if changed:
                break

    return kept, dropped, mi_norm


def print_and_plot_selection(kept, dropped, mi_norm, mi_vs_fesc,
                              spearman_full_df, out_path,
                              target="fesc_dir",
                              mi_threshold=0.3, redundancy_threshold=0.8):
    """
    Print a clear summary of what was kept/dropped and why,
    then plot the final Spearman sub-matrix of kept indicators + fesc_dir.
    """
    print(f"\n  Step 2 – Redundancy removal (|Spearman rho| >= {redundancy_threshold},")
    print(f"           drop the partner with lower MI vs fesc):")
    if not dropped:
        print("    No redundant pairs found — all candidates kept.")
    for loser, winner, rho, mi_l, mi_w in dropped:
        print(f"    DROP {loser:<14} (MI_norm={mi_l:.3f})  "
              f"<->  KEEP {winner:<14} (MI_norm={mi_w:.3f})  |rho|={rho:.3f}")

    print(f"\n  Final recommended indicator set ({len(kept)}):")
    for ind in kept:
        print(f"    {ind:<14}  MI_norm={mi_norm[ind]:.3f}")

    if not HAS_MPL or not kept:
        return

    # plot Spearman sub-matrix of final set + fesc_dir
    cols_sub  = kept + [target]
    spear_sub = spearman_full_df.loc[cols_sub, cols_sub]
    n         = len(cols_sub)

    fig, ax = plt.subplots(figsize=(max(5, n * 0.9), max(4, n * 0.85)))
    im   = ax.imshow(spear_sub.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho", fontsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols_sub, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols_sub, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = spear_sub.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    # highlight fesc_dir row/column in gold
    fesc_idx = cols_sub.index(target)
    for pos in [fesc_idx - 0.5, fesc_idx + 0.5]:
        ax.axhline(pos, color="gold", linewidth=2.5)
        ax.axvline(pos, color="gold", linewidth=2.5)

    ax.set_title(
        f"Final non-redundant indicators – Spearman co-correlation\n"
        f"(MI_norm >= {mi_threshold}, |rho| redundancy cutoff = {redundancy_threshold};  gold = fesc_dir)",
        fontsize=10, pad=10
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ─────────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading simulation: {SIM_CATALOGUE}")
    sim_raw = pd.read_csv(SIM_CATALOGUE)
    print(f"      Rows: {len(sim_raw):,}   Columns: {sim_raw.shape[1]}")

    # ── extract indicators ────────────────────────────────────────────────────
    print(f"\n[2/5] Extracting indicators ({N_DIRS} directions × {len(sim_raw):,} halos)…")
    sim_ind = extract_sim_directional(sim_raw, n_dirs=N_DIRS)
    print(f"      Expanded rows: {len(sim_ind):,}  (each row = one independent galaxy)")
    sim_ind.to_csv(out_dir / "sim_indicators.csv", index=False)
    print(f"      Saved: {out_dir / 'sim_indicators.csv'}")
    print_summary(sim_ind, ALL_COLS, len(sim_ind[ALL_COLS].dropna()))

    # ── Pearson ───────────────────────────────────────────────────────────────
    print(f"\n[3/5] Computing Pearson correlation matrix…")
    pearson_corr, n_p = compute_pearson(sim_ind, ALL_COLS)
    pearson_corr.to_csv(out_dir / "sim_pearson_matrix.csv")
    print(f"      n={n_p:,}  |  Saved: sim_pearson_matrix.csv")

    # ── Spearman ──────────────────────────────────────────────────────────────
    print(f"\n[4/5] Computing Spearman correlation matrix…")
    spearman_corr, n_s = compute_spearman(sim_ind, ALL_COLS)
    spearman_corr.to_csv(out_dir / "sim_spearman_matrix.csv")
    print(f"      n={n_s:,}  |  Saved: sim_spearman_matrix.csv")

    # ── Mutual Information ────────────────────────────────────────────────────
    print(f"\n[5/5] Computing Mutual Information matrix (may take a moment)…")
    mi_matrix, mi_vs_fesc, n_mi = compute_mutual_info(sim_ind, ALL_COLS)
    mi_matrix.to_csv(out_dir  / "sim_mi_matrix.csv")
    mi_vs_fesc.to_csv(out_dir / "sim_mi_vs_fesc.csv", header=True)
    print(f"      n={n_mi:,}  |  Saved: sim_mi_matrix.csv, sim_mi_vs_fesc.csv")

    # ── ranking table ─────────────────────────────────────────────────────────
    ranking = build_ranking_table(pearson_corr, spearman_corr, mi_vs_fesc)
    ranking.to_csv(out_dir / "sim_indicator_ranking.csv")
    print_ranking(ranking)

    # ── heatmaps + bar chart ──────────────────────────────────────────────────
    print(f"\nSaving plots…")
    plot_heatmap(pearson_corr,  "SPHINX20 – Pearson Correlation",
                 str(out_dir / "sim_pearson_heatmap.png"),
                 cmap="RdBu_r", vmin=-1, vmax=1)

    plot_heatmap(spearman_corr, "SPHINX20 – Spearman Correlation",
                 str(out_dir / "sim_spearman_heatmap.png"),
                 cmap="RdBu_r", vmin=-1, vmax=1)

    mi_norm_factor = float(mi_vs_fesc.max() + 1e-30)
    mi_matrix_norm = mi_matrix / mi_norm_factor
    plot_heatmap(mi_matrix_norm, "SPHINX20 – Mutual Information (normalised by max MI vs fesc)",
                 str(out_dir / "sim_mi_heatmap.png"),
                 cmap="YlOrRd", vmin=0, vmax=1,
                 fmt=".3f", annot_thresh=0.7)

    plot_ranking_bar(ranking, str(out_dir / "sim_indicator_ranking.png"))

    # ── indicator selection: MI filter + redundancy removal ──────────────────
    print("\n[6/6] Selecting final non-redundant indicators...")
    kept, dropped, mi_norm_sel = select_final_indicators(
        mi_vs_fesc, spearman_corr,
        mi_threshold=0.3, redundancy_threshold=0.8
    )
    print_and_plot_selection(
        kept, dropped, mi_norm_sel, mi_vs_fesc, spearman_corr,
        str(out_dir / "sim_final_indicators.png"),
        mi_threshold=0.3, redundancy_threshold=0.8
    )

    print(f"\n✓ Done. All outputs in: {out_dir.resolve()}\n")

    # ── observations (commented out for now) ─────────────────────────────────
    # obs_raw = pd.read_csv(OBS_CATALOGUE)
    # obs_ind = extract_obs(obs_raw)
    # pearson_obs,  _ = compute_pearson(obs_ind, ALL_COLS)
    # spearman_obs, _ = compute_spearman(obs_ind, ALL_COLS)
    # mi_obs, mi_vs_fesc_obs, _ = compute_mutual_info(obs_ind, ALL_COLS)