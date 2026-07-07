import argparse
import csv
import os
import random
from collections import defaultdict

METRICS = ["last_test_return", "train_auc", "best_train_return", "stability_gap"]
BUDGET = ["last_llm_fd_records", "last_llm_fd_shaping_triggers", "last_llm_fd_shaping_penalty_total", "last_llm_fd_shaping_terminal_bonus_total", "last_llm_fd_shaping_episode_steps_total", "last_llm_fd_shaping_avg_penalty_per_trigger", "last_llm_fd_shaping_avg_steps_per_trigger"]


def load_rows(path):
    if not os.path.exists(path): return []
    with open(path, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def parse(row):
    name=row.get('method','')
    if name.startswith('smallcoop_'):
        prefix='smallcoop_lbforaging_Foraging-8x8-2p-2f-coop-v3_'
        return 'smallcoop', name[len(prefix):] if name.startswith(prefix) else name
    if name.startswith('maincoop_'):
        prefix='maincoop_lbforaging_Foraging-10x10-3p-3f-coop-v3_'
        return 'maincoop', name[len(prefix):] if name.startswith(prefix) else name
    if name.startswith('scale15_'):
        prefix='scale15_lbforaging_Foraging-15x15-3p-4f-v3_'
        return 'scale15', name[len(prefix):] if name.startswith(prefix) else name
    return 'other', name

def f(x):
    try:
        if x in {'', None}: return None
        return float(x)
    except (TypeError, ValueError): return None

def mean(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None

def ci(xs,n=10000,seed=1212):
    xs=[x for x in xs if x is not None]
    if not xs: return None,None
    rng=random.Random(seed); vals=[]
    for _ in range(n): vals.append(sum(xs[rng.randrange(len(xs))] for _ in xs)/len(xs))
    vals.sort(); return vals[int(.025*n)], vals[int(.975*n)]

def fmt(x): return 'NA' if x is None else f'{x:.4f}'

def write_group(h, rows, group, title):
    by=defaultdict(list)
    for r in rows:
        g,m=parse(r)
        if g==group: by[m].append(r)
    if not by: return
    h.write(f'## {title}\n\n')
    h.write('| method | n | last test | last test 95% CI | train AUC | best train | stability gap |\n')
    h.write('|---|---:|---:|---:|---:|---:|---:|\n')
    for m in sorted(by):
        vals={k:[f(r.get(k)) for r in by[m]] for k in METRICS}
        lo,hi=ci(vals['last_test_return'])
        h.write(f"| {m} | {len(by[m])} | {fmt(mean(vals['last_test_return']))} | [{fmt(lo)}, {fmt(hi)}] | {fmt(mean(vals['train_auc']))} | {fmt(mean(vals['best_train_return']))} | {fmt(mean(vals['stability_gap']))} |\n")
    h.write('\n')
    h.write(f'## {title} Paired Comparisons\n\n')
    h.write('| comparison | metric | n | mean delta | 95% CI | seeds |\n')
    h.write('|---|---|---:|---:|---:|---|\n')
    seedmap={m:{r['seed']:r for r in rs} for m,rs in by.items()}
    for right in ['baseline','uniform_budget_matched_0.0003_late045','random_type_budget_matched_0.0003_late045']:
        if 'adaptive_0.0003_late045' not in seedmap or right not in seedmap: continue
        seeds=sorted(set(seedmap['adaptive_0.0003_late045']) & set(seedmap[right]), key=lambda x:int(x) if str(x).isdigit() else str(x))
        for met in METRICS:
            ds=[]
            for s in seeds:
                av=f(seedmap['adaptive_0.0003_late045'][s].get(met)); rv=f(seedmap[right][s].get(met))
                if av is not None and rv is not None: ds.append(av-rv)
            lo,hi=ci(ds)
            h.write(f"| adaptive - {right} | {met} | {len(ds)} | {fmt(mean(ds))} | [{fmt(lo)}, {fmt(hi)}] | {','.join(seeds)} |\n")
    h.write('\n')
    h.write(f'## {title} Budget Accounting\n\n')
    h.write('| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |\n')
    h.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for m in sorted(by):
        vals={k:[f(r.get(k)) for r in by[m]] for k in BUDGET}
        h.write(f"| {m} | {fmt(mean(vals['last_llm_fd_records']))} | {fmt(mean(vals['last_llm_fd_shaping_triggers']))} | {fmt(mean(vals['last_llm_fd_shaping_penalty_total']))} | {fmt(mean(vals['last_llm_fd_shaping_terminal_bonus_total']))} | {fmt(mean(vals['last_llm_fd_shaping_episode_steps_total']))} | {fmt(mean(vals['last_llm_fd_shaping_avg_penalty_per_trigger']))} | {fmt(mean(vals['last_llm_fd_shaping_avg_steps_per_trigger']))} |\n")
    h.write('\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir', required=True); args=ap.parse_args()
    rows=load_rows(os.path.join(args.out_dir,'summary.csv'))
    with open(os.path.join(args.out_dir,'ROUND12_SIMPLE_DOMAIN_REPORT.md'),'w',encoding='utf-8') as h:
        h.write('# Round 12 Simple Cooperative Domain Report\n\n')
        h.write('Round 12 searches for simple sparse cooperative domains that can provide positive family-level evidence without relying on VMAS/RWARE/Qwen.\n\n')
        write_group(h, rows, 'smallcoop', 'LBF 8x8 2p-2f Coop')
        write_group(h, rows, 'maincoop', 'LBF 10x10 3p-3f Coop')
        write_group(h, rows, 'scale15', 'LBF 15x15 3p-4f')
        h.write('## Decision Rule\n\nUse a domain as positive evidence only if adaptive beats baseline and remains competitive with or better than phase-uniform/random controls under paired seeds. Otherwise report it as a stress test.\n')
if __name__=='__main__': main()
