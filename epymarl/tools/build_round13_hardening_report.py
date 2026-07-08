import argparse
import csv
import os
import random
from collections import defaultdict

METRICS = ["last_test_return", "train_auc", "best_train_return", "stability_gap"]
BUDGET = ["last_llm_fd_records", "last_llm_fd_shaping_triggers", "last_llm_fd_shaping_penalty_total", "last_llm_fd_shaping_terminal_bonus_total", "last_llm_fd_shaping_episode_steps_total", "last_llm_fd_shaping_avg_penalty_per_trigger", "last_llm_fd_shaping_avg_steps_per_trigger"]
PREFIXES = [
    "seedext10_lbforaging_Foraging-10x10-3p-3f-v3_",
    "actualbudget10_lbforaging_Foraging-10x10-3p-3f-v3_",
    "sensitivity10ext_lbforaging_Foraging-10x10-3p-3f-v3_",
]


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def parse_method(name):
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return prefix.split('_')[0], name[len(prefix):]
    return 'other', name


def f(x):
    try:
        if x in {'', None}:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ci(xs, n=10000, seed=1313):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs))
    vals.sort()
    return vals[int(0.025*n)], vals[int(0.975*n)]


def fmt(x):
    return 'NA' if x is None else f'{x:.4f}'


def group_rows(rows, group):
    out = defaultdict(list)
    for r in rows:
        g, m = parse_method(r.get('method', ''))
        if g == group:
            out[m].append(r)
    return out


def write_table(h, title, by):
    if not by:
        return
    h.write(f'## {title}\n\n')
    h.write('| method | n | last test | last test 95% CI | train AUC | best train | stability gap |\n')
    h.write('|---|---:|---:|---:|---:|---:|---:|\n')
    for m in sorted(by):
        vals = {k: [f(r.get(k)) for r in by[m]] for k in METRICS}
        lo, hi = ci(vals['last_test_return'])
        h.write(f"| {m} | {len(by[m])} | {fmt(mean(vals['last_test_return']))} | [{fmt(lo)}, {fmt(hi)}] | {fmt(mean(vals['train_auc']))} | {fmt(mean(vals['best_train_return']))} | {fmt(mean(vals['stability_gap']))} |\n")
    h.write('\n')


def write_budget(h, title, by):
    if not by:
        return
    h.write(f'## {title} Budget Accounting\n\n')
    h.write('| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |\n')
    h.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for m in sorted(by):
        vals = {k: [f(r.get(k)) for r in by[m]] for k in BUDGET}
        h.write(f"| {m} | {fmt(mean(vals['last_llm_fd_records']))} | {fmt(mean(vals['last_llm_fd_shaping_triggers']))} | {fmt(mean(vals['last_llm_fd_shaping_penalty_total']))} | {fmt(mean(vals['last_llm_fd_shaping_terminal_bonus_total']))} | {fmt(mean(vals['last_llm_fd_shaping_episode_steps_total']))} | {fmt(mean(vals['last_llm_fd_shaping_avg_penalty_per_trigger']))} | {fmt(mean(vals['last_llm_fd_shaping_avg_steps_per_trigger']))} |\n")
    h.write('\n')


def write_pairs(h, title, by, left='adaptive_0.0003_late045', rights=None):
    if rights is None:
        rights = ['baseline', 'uniform_budget_matched_0.0003_late045', 'random_type_budget_matched_0.0003_late045', 'uniform_actual_budget_matched', 'random_actual_budget_matched']
    seedmap = {m: {r['seed']: r for r in rs} for m, rs in by.items()}
    if left not in seedmap:
        return
    h.write(f'## {title} Paired Comparisons\n\n')
    h.write('| comparison | metric | n | mean delta | 95% CI | seeds |\n')
    h.write('|---|---|---:|---:|---:|---|\n')
    for right in rights:
        if right not in seedmap:
            continue
        seeds = sorted(set(seedmap[left]) & set(seedmap[right]), key=lambda x: int(x) if str(x).isdigit() else str(x))
        for metric in METRICS:
            ds = []
            for s in seeds:
                lv = f(seedmap[left][s].get(metric)); rv = f(seedmap[right][s].get(metric))
                if lv is not None and rv is not None:
                    ds.append(lv - rv)
            lo, hi = ci(ds)
            h.write(f'| {left} - {right} | {metric} | {len(ds)} | {fmt(mean(ds))} | [{fmt(lo)}, {fmt(hi)}] | {",".join(seeds)} |\n')
    h.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    rows = load(os.path.join(args.out_dir, 'summary.csv'))
    with open(os.path.join(args.out_dir, 'ROUND13_HARDENING_REPORT.md'), 'w', encoding='utf-8') as h:
        h.write('# Round 13 AAAI Main-Claim Hardening Report\n\n')
        h.write('Round 13 strengthens the main 10x10 claim by adding seeds 9-16, actual-budget matched controls, and sensitivity extensions.\n\n')
        for group, title in [('seedext10', '10x10 Seed Extension'), ('actualbudget10', '10x10 Actual-Budget Controls'), ('sensitivity10ext', '10x10 Sensitivity Extension')]:
            by = group_rows(rows, group)
            write_table(h, title, by)
            write_pairs(h, title, by)
            write_budget(h, title, by)
        h.write('## Decision Rule\n\n')
        h.write('Use the main claim as AAAI-ready only if adaptive remains positive against baseline, phase-uniform, random-type, and actual-budget controls when merged with prior 10x10 evidence. Sensitivity variants should show that the adaptive family is not a single fragile point.\n')

if __name__ == '__main__':
    main()
