"""
cluster_ci.py -- Cluster-robust confidence intervals (review B3).

Reviewer blocker B3: the paper's headline CIs (e.g. P8 [2.377, 2.438]) are
"matrix-clustered bootstrap" over as few as G=4 matrices, reported to three
decimals. A percentile bootstrap over 4 clusters badly under-covers, so those
widths overstate precision. B3 asks to (a) report the cluster count G beside
every CI and (b) re-estimate with a small-cluster-robust method (wild cluster
bootstrap, or >= 20-30 clusters).

This module provides three 0-GPU estimators plus a coverage simulation:
  * cluster_bootstrap_ci      : ordinary cluster (block) percentile bootstrap
  * wild_cluster_bootstrap_ci : Rademacher wild cluster bootstrap-t (CGM 2008),
                                the recommended small-G fix
  * cluster_t_ci              : cluster-robust SE with a Student-t(G-1) crit val
All take a per-observation value array x and an integer group label per obs.

The public API also includes recompute_from_archive(), a CLI that reads the
paper's repro archive (per-cell metrics.json / CSV with columns
[family, matrix_id, seed, effect]) and regenerates every CI with all three
methods and the cluster count G -- this is the exact 0-GPU recomputation to run
on the authors' archived data (which was NOT included in the upload).
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
def _group_means(x, groups):
    g = np.asarray(groups)
    uniq = np.unique(g)
    return np.array([x[g == u].mean() for u in uniq]), uniq


def cluster_bootstrap_ci(x, groups, n_boot=20000, alpha=0.05, seed=0):
    """Percentile cluster bootstrap: resample whole clusters with replacement,
    recompute the overall (obs-weighted) mean. This mirrors the paper's stated
    'matrix-clustered bootstrap'. Under-covers when the number of clusters G is
    small -- that is exactly what the coverage sim below demonstrates.

    Vectorized for the common balanced case (all clusters equal size, incl.
    singleton clusters -> ordinary iid bootstrap); falls back to a loop for
    unbalanced clusters."""
    x = np.asarray(x, float); g = np.asarray(groups)
    uniq = np.unique(g)
    G = len(uniq)
    idx_by_group = [np.where(g == u)[0] for u in uniq]
    sizes = np.array([len(ix) for ix in idx_by_group])
    rng = np.random.default_rng(seed)
    if np.all(sizes == sizes[0]):
        # balanced: cluster means suffice; resample G cluster means with repl.
        cmeans = np.array([x[ix].mean() for ix in idx_by_group])
        pick = rng.integers(0, G, size=(n_boot, G))
        boots = cmeans[pick].mean(axis=1)
    else:
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, G, size=G)
            idx = np.concatenate([idx_by_group[p] for p in pick])
            boots[b] = x[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def cluster_t_ci(x, groups, alpha=0.05):
    """Cluster-robust SE of the mean with a t(G-1) critical value.
    SE^2 = sum_g (n_g/N)^2 * Var_between-approx; here we use the standard
    cluster-mean variance:  Var(xbar) = (1/G) * sample var of cluster means / G-ish.
    We use the simple, well-behaved cluster-mean estimator:
        xbar = mean of cluster means (equal cluster weights)
        SE   = sd(cluster means) / sqrt(G)
    with Student-t(G-1). Equal-cluster weighting is appropriate when clusters
    are balanced (as in the paper's designs)."""
    from scipy import stats
    means, uniq = _group_means(np.asarray(x, float), groups)
    G = len(means)
    xbar = means.mean()
    se = means.std(ddof=1) / np.sqrt(G)
    tcrit = stats.t.ppf(1 - alpha / 2, df=G - 1)
    return float(xbar - tcrit * se), float(xbar + tcrit * se)


def wild_cluster_bootstrap_ci(x, groups, n_boot=20000, alpha=0.05, seed=0):
    """Rademacher wild cluster bootstrap-t for the mean (Cameron, Gelbach &
    Miller 2008) -- the recommended estimator when G is small. We bootstrap the
    cluster-robust t-statistic for H0: mu = xbar_hat under Rademacher (+/-1)
    cluster weights and invert to a percentile-t interval.
    """
    x = np.asarray(x, float)
    means, uniq = _group_means(x, groups)
    G = len(means)
    xbar = means.mean()
    se = means.std(ddof=1) / np.sqrt(G)
    if se == 0:
        return float(xbar), float(xbar)
    resid = means - xbar
    rng = np.random.default_rng(seed)
    W = rng.choice([-1.0, 1.0], size=(n_boot, G))     # Rademacher weights
    star = xbar + W * resid                            # (n_boot, G) wild means
    sb = star.std(axis=1, ddof=1) / np.sqrt(G)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstars = np.where(sb > 0, (star.mean(axis=1) - xbar) / sb, 0.0)
    qlo, qhi = np.quantile(tstars, [alpha / 2, 1 - alpha / 2])
    # percentile-t: invert  (xbar - mu)/se  in  [qlo, qhi]
    return float(xbar - qhi * se), float(xbar - qlo * se)


def all_cis(x, groups, seed=0):
    G = len(np.unique(groups))
    return dict(
        G=G, mean=float(np.mean(x)),
        percentile_cluster=cluster_bootstrap_ci(x, groups, seed=seed),
        wild_cluster=wild_cluster_bootstrap_ci(x, groups, seed=seed),
        cluster_t=cluster_t_ci(x, groups),
    )


# ----------------------------------------------------------------------------
# Coverage simulation: demonstrates the small-G under-coverage and the fix
# ----------------------------------------------------------------------------
def coverage_sim(G, n_per=10, mu=2.4, between_sd=0.05, within_sd=0.4,
                 reps=2000, seed=0):
    """Monte-Carlo coverage of a nominal 95% CI for the mean when data are
    clustered in G groups (matrices) with n_per obs (seeds) each. Returns the
    empirical coverage of each estimator (target 0.95)."""
    rng = np.random.default_rng(seed)
    hit = {"percentile_cluster": 0, "wild_cluster": 0, "cluster_t": 0}
    width = {"percentile_cluster": 0.0, "wild_cluster": 0.0, "cluster_t": 0.0}
    for r in range(reps):
        clus = rng.normal(0, between_sd, size=G)
        x = []; grp = []
        for gi in range(G):
            x.append(rng.normal(mu + clus[gi], within_sd, size=n_per))
            grp.append(np.full(n_per, gi))
        x = np.concatenate(x); grp = np.concatenate(grp)
        for name, fn in (("percentile_cluster", cluster_bootstrap_ci),
                         ("wild_cluster", wild_cluster_bootstrap_ci),
                         ("cluster_t", cluster_t_ci)):
            if name == "cluster_t":
                lo, hi = fn(x, grp)
            else:
                lo, hi = fn(x, grp, n_boot=2000, seed=r)
            hit[name] += int(lo <= mu <= hi)
            width[name] += (hi - lo)
    return {k: (hit[k] / reps, width[k] / reps) for k in hit}


def recompute_from_archive(path, effect_col="effect", family_col="family",
                           matrix_col="matrix_id"):
    """CLI helper: read the authors' repro archive and regenerate CIs.
    Accepts CSV or JSON-lines with columns [family, matrix_id, seed, effect].
    Prints, per family, mean and all three CIs with the cluster count G.
    (Run this on the archived metrics data -- 0 GPU.)"""
    import csv, json, os, collections
    recs = collections.defaultdict(lambda: ([], []))  # family -> (effects, mats)
    if path.endswith(".json") or path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                recs[d[family_col]][0].append(float(d[effect_col]))
                recs[d[family_col]][1].append(d[matrix_col])
    else:
        with open(path) as f:
            for d in csv.DictReader(f):
                recs[d[family_col]][0].append(float(d[effect_col]))
                recs[d[family_col]][1].append(d[matrix_col])
    print(f"# cluster-robust recomputation from {os.path.basename(path)}")
    for fam, (eff, mats) in recs.items():
        x = np.array(eff)
        # map matrix ids to integer group labels
        uniq = {m: i for i, m in enumerate(dict.fromkeys(mats))}
        grp = np.array([uniq[m] for m in mats])
        out = all_cis(x, grp)
        print(f"\n[{fam}]  n_obs={len(x)}  G(clusters)={out['G']}  "
              f"mean={out['mean']:+.3f}")
        print(f"   percentile-cluster 95% CI : [{out['percentile_cluster'][0]:+.3f}, "
              f"{out['percentile_cluster'][1]:+.3f}]  (paper's method)")
        print(f"   wild-cluster       95% CI : [{out['wild_cluster'][0]:+.3f}, "
              f"{out['wild_cluster'][1]:+.3f}]  (recommended for small G)")
        print(f"   cluster-t(G-1)     95% CI : [{out['cluster_t'][0]:+.3f}, "
              f"{out['cluster_t'][1]:+.3f}]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        recompute_from_archive(sys.argv[1])
        sys.exit(0)
    print("=" * 74)
    print("COVERAGE of a nominal 95% CI vs. number of clusters G (target 0.95)")
    print("data: G matrices x 10 seeds, between-matrix sd 0.05, within 0.40")
    print("=" * 74)
    print(f"{'G':>4} | {'percentile-cluster':>22} | {'wild-cluster':>18} | {'cluster-t(G-1)':>18}")
    print("-" * 74)
    for G in (4, 8, 16, 30):
        cov = coverage_sim(G, reps=1500, seed=G)
        def c(k): return f"{cov[k][0]*100:5.1f}%  w={cov[k][1]:.3f}"
        print(f"{G:>4} | {c('percentile_cluster'):>22} | "
              f"{c('wild_cluster'):>18} | {c('cluster_t'):>18}")
    print("\nReading: at G=4 the percentile-cluster interval (the paper's method)")
    print("under-covers (well below 95%); wild-cluster / cluster-t restore")
    print("coverage at the cost of honestly wider intervals.")
