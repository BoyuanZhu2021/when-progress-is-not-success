"""Export the manuscript's fixed cohorts from existing runs; never run training.

Uses the project's paired-test implementation. Figures can subsequently be
rebuilt from results/paper_data.json without access to the research repository.
"""
import hashlib
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'code' / 'scripts'))
from a2_analyze import paired_t, bootstrap_ci, t_two_sided_p

SOURCES = {}

def audit_training_traces():
    """Read existing original-campaign traces; do not run or rescore any model.

    Reconstruct the archived training reward trigger from the saved anchor
    and m_train, then compare it with the saved strict security outcome.
    This checks implementation semantics, not the correctness of raw-text
    verifiers. Missing count traces used chain progress in the RL trigger.
    """
    snapshot = json.loads((PAPER/'results/paper_data.json').read_text(encoding='utf-8'))
    audit = {'scope':'original campaign training records only',
             'limitations':'Uses saved phi/count traces and security flags; not independent raw-text verification.',
             'code_reference':'63f44b4:code/scripts/a3_multiturn_train.py::_anchor and _succ_m',
             'runs':{}, 'by_arm':{}, 'sources':{}}
    for arm, seeds in snapshot['fullft'].items():
        by_count_availability = {}
        for seed, run in seeds.items():
            path = ROOT/run['run']/'turns.jsonl'
            digest = hashlib.sha256()
            n = 0
            with path.open('rb') as handle:
                for line in handle:
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get('split') or 'phi_trace' not in row:
                        continue
                    n += 1
                    count = row.get('count_trace') or []
                    # Numeric task-ID ranges overlap across domains. Group by
                    # the recorded field availability rather than guessing domain.
                    category = 'count_present' if count else 'count_missing'
                    stats = by_count_availability.setdefault(category, dict(records=0, missing_count=0,
                        anchor_positive=0, strict_success=0, anchor_vs_strict_disagree=0,
                        anchor_positive_strict_failure=0, anchor_negative_strict_success=0,
                        count_vs_strict_disagree=0))
                    anchor = count or row['phi_trace']
                    m = row.get('m_train') or 4
                    fired = max(anchor,default=0) >= m/4
                    strict = bool(row['success'])
                    stats['records'] += 1
                    stats['missing_count'] += not bool(count)
                    stats['anchor_positive'] += fired
                    stats['strict_success'] += strict
                    stats['anchor_vs_strict_disagree'] += fired != strict
                    stats['anchor_positive_strict_failure'] += fired and not strict
                    stats['anchor_negative_strict_success'] += strict and not fired
                    stats['count_vs_strict_disagree'] += bool(count) and ((max(count) >= 1-1e-9) != strict)
            audit['runs'][run['run']] = n
            audit['sources'][str(path.relative_to(ROOT)).replace('\\','/')] = digest.hexdigest()
        audit['by_arm'][arm] = by_count_availability
    target = PAPER/'results/proofread_training_audit.json'
    target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit['by_arm'],indent=2))
    print('Audited',sum(audit['runs'].values()),'records in',len(audit['runs']),'existing runs.')

def audit_eval_traces():
    """Compare both archived terminal checks on the complete dynamics cohort."""
    data = json.loads((PAPER/'results/paper_data.json').read_text(encoding='utf-8'))
    out = {'scope':'complete dynamics cohort, saved evaluation episodes',
           'limitations':'Compares stored checks, not an independent semantic rescore.',
           'runs':{},'sources':{}}
    for arm, seeds in data['dynamics'].items():
        for seed, run in seeds.items():
            path=ROOT/run['run']/'eval_turns.jsonl'
            digest=hashlib.sha256(); groups={}
            with path.open('rb') as handle:
                for line in handle:
                    digest.update(line)
                    if not line.strip(): continue
                    row=json.loads(line)
                    if str(row['step']) not in run['eval']: continue
                    key=f"{row['step']}_{row['split']}"
                    cell=groups.setdefault(key,{'goals':{},'records':0,'disagree':0,'count_missing':0})
                    count=row.get('count_trace') or []
                    sec=bool(row['success'])
                    logged=(max(count)>=1-1e-9) if count else sec
                    g=cell['goals'].setdefault(row['goal'],{'n':0,'logged':0,'security':0})
                    g['n']+=1;g['logged']+=logged;g['security']+=sec
                    cell['records']+=1;cell['disagree']+=logged!=sec
                    cell['count_missing']+=not bool(count)
            for key, cell in groups.items():
                gs=cell.pop('goals')
                cell['goal_ids']=len(gs)
                cell['logged_asr']=float(np.mean([g['logged']/g['n'] for g in gs.values()]))
                cell['security_asr']=float(np.mean([g['security']/g['n'] for g in gs.values()]))
                step,split=key.split('_',1)
                metric={'ood':'ood_asr','indomain':'indomain_asr','transfer':'transfer_asr'}[split]
                assert abs(cell['logged_asr']-run['eval'][step][metric])<=5.01e-5, (arm,seed,key)
            out['runs'][run['run']]=groups
            out['sources'][str(path.relative_to(ROOT)).replace('\\','/')]=digest.hexdigest()
    (PAPER/'results/proofread_eval_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    for split in ('ood','indomain','transfer'):
        cells=[c for rs in out['runs'].values() for k,c in rs.items() if k.endswith('_'+split)]
        print(split,'episodes',sum(c['records'] for c in cells),'disagreements',sum(c['disagree'] for c in cells),
              'max ASR gap',max(abs(c['logged_asr']-c['security_asr']) for c in cells))

def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8-sig')

def load_campaign(folder, arms, steps):
    result = {}
    for arm, seeds in arms.items():
        result[arm] = {}
        for seed in seeds:
            run = ROOT / 'artifacts' / folder / f's{seed}_{arm}'
            rows = [json.loads(x) for x in read(run / 'progress.jsonl').splitlines() if x.strip()]
            meta = json.loads(read(run / 'run_meta.json'))
            ev = {str(r['step']): r for r in rows if r.get('eval') and r['step'] in steps}
            assert set(ev) == set(map(str, steps)), (run, sorted(ev), steps)
            tr = [r for r in rows if not r.get('eval') and 1 <= r['step'] <= max(steps)]
            assert len(tr) == max(steps), (run, len(tr))
            metrics = ['all_zero_group_frac', 'train_mean_phi', 'train_success', 'n_examples', 'step_time']
            means = {k: float(np.mean([r[k] for r in tr if r.get(k) is not None]))
                     for k in metrics if all(r.get(k) is not None for r in tr)}
            # Per-iteration series for the training-dynamics figures. Values are
            # copied from the recorded progress rows; missing entries stay null.
            tr_by_step = {r['step']: r for r in tr}
            series_metrics = ['train_success', 'train_mean_phi', 'all_zero_group_frac',
                              'frac_zero_adv', 'n_examples', 'pg_loss', 'kl_loss', 'grad_norm']
            series = {'steps': list(range(1, max(steps)+1)),
                      **{k: [tr_by_step[s].get(k) if s in tr_by_step else None
                             for s in range(1, max(steps)+1)] for k in series_metrics}}
            # Every recorded evaluation checkpoint up to the last read point, not only
            # the fixed read points used by the tables.
            eval_all = {str(r['step']): {m: r[m] for m in ['ood_asr', 'ood_phi', 'indomain_asr',
                        'transfer_asr'] if m in r} for r in rows
                        if r.get('eval') and r['step'] <= max(steps)}
            result[arm][str(seed)] = {'run': str(run.relative_to(ROOT)).replace('\\', '/'),
                'eval': {k: {m: r[m] for m in ['ood_asr', 'ood_phi', 'indomain_asr', 'transfer_asr']
                              if m in r} for k, r in ev.items()}, 'train_mean': means,
                'series': series, 'eval_all': eval_all,
                'config': {k: meta.get(k) for k in ['G', 'T', 'K', 'lr', 'baseline', 'full_ft',
                            'beta_kl_effective', 'eval_every', 'n_train_goals', 'n_ood_goals']}}
    return result

def contrast(campaign, arm, base, step, metric='ood_asr'):
    seeds = sorted(set(campaign[arm]) & set(campaign[base]), key=int)
    a = {s: campaign[arm][s]['eval'][str(step)] for s in seeds}
    b = {s: campaign[base][s]['eval'][str(step)] for s in seeds}
    test = paired_t(a, b, seeds, metric)
    boot = bootstrap_ci(a, b, seeds, metric)
    diffs = np.array([a[s][metric] - b[s][metric] for s in seeds])
    low, high = 0.0, 100.0
    for _ in range(70):
        mid = (low + high) / 2
        if t_two_sided_p(mid, len(seeds)-1) > .05:
            low = mid
        else:
            high = mid
    half = high * diffs.std(ddof=1) / np.sqrt(len(seeds))
    return {'seeds': seeds, 'arm_mean': float(np.mean([a[s][metric] for s in seeds])),
            'base_mean': float(np.mean([b[s][metric] for s in seeds])), **test,
            't_ci': [float(diffs.mean()-half), float(diffs.mean()+half)], 'bootstrap': boot}

def main():
    full = load_campaign('fullft_campaign', {'sparse': range(2,18), 'dense':range(2,8),
         'hindsight':range(2,7), 'curriculum':range(2,8), 'vip':range(2,6),
         'sil':range(2,6), 'G12':range(2,9), 'sft':range(2,18)}, list(range(2,17,2)))
    dyn = load_campaign('sft_dynamics', {'sparse':range(2,9), 'sft':range(2,9),
              'rl_pos':range(2,8), 'sft_all':[2]}, [8,16,24,32])
    scale = load_campaign('scale_9b_27b38_phase1', {'sparse':range(2,6),
                          'dense':range(2,6), 'sft':range(2,6)}, [8,16])
    expansion = {label: contrast({a:{s:r for s,r in full[a].items() if int(s) in seeds}
                     for a in ['sft','sparse']},'sft','sparse',16)
                 for label,seeds in [('initial_8',range(2,10)),('added_8',range(10,18))]}
    published = {f: json.loads(read(ROOT/'docs/results/sft-dynamics-4b-9b'/f'{f}.json'))
                 for f in ['_T12-full','_T12-1632-full','_T3-1632-full','_side-full','_side-1632-full']}
    data = {'version':2, 'created':'2026-09-11', 'fullft':full, 'dynamics':dyn, 'scale':scale,
        'fullft_comparisons':{a:contrast(full,a,'sparse',16) for a in full if a!='sparse'},
        'fullft_gap':{str(s):contrast(full,'sft','sparse',s) for s in range(2,17,2)},
        'published_dynamics':published, 'ofc_expansion':expansion,
        # Paired-t contrasts for the mechanism arms at every read point, so the
        # tables can use one interval type throughout; the saved bootstrap
        # intervals in published_dynamics remain the reported ones in the text.
        'dynamics_comparisons':{a:{str(s):contrast(dyn,a,'sparse',s) for s in [8,16,24,32]}
                                for a in ['sft','rl_pos']},
        'scale_comparisons':{a:contrast(scale,a,'sparse',16) for a in ['dense','sft']},
        'scale_comparisons_by_step':{a:{str(s):contrast(scale,a,'sparse',s) for s in [8,16]}
                                     for a in ['dense','sft']},
        'transfer_comparisons':{f'{c}_{a}':contrast(rows,a,'sparse',step,'transfer_asr')
             for c,rows,step,arms in [('fullft',full,16,['dense','sft']),
                 ('dynamics',dyn,32,['sft','rl_pos']),('scale',scale,16,['dense','sft'])] for a in arms}}
    # Recorded contrasts are authoritative for published bootstrap intervals;
    # cross-check means against the newly exported raw trajectory summaries.
    for label, arm in [('_T12-full','sft'),('_side-full','rl_pos')]:
        for step, row in published[label]['per_step'].items():
            actual=contrast(dyn,arm,'sparse',int(step))
            assert abs(actual['effect']-row['paired_t']['effect']) < 2e-6
    assert abs(data['fullft_comparisons']['dense']['effect'] + .1400) < .0001
    assert abs(data['fullft_comparisons']['sft']['effect'] - .0460) < .0001
    out = PAPER/'results'
    out.mkdir(exist_ok=True)
    (out/'paper_data.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    (out/'sources_manifest.json').write_text(json.dumps(SOURCES,indent=2)+'\n',encoding='utf-8')
    for arm in ['sparse','dense']:
        selected = [full[arm][str(s)] for s in range(2,8)]
        print(arm, {k: round(float(np.mean([r['train_mean'][k] for r in selected])),5)
                    for k in ['all_zero_group_frac','n_examples','train_mean_phi']},
                    'OOD Phi',np.mean([r['eval']['16']['ood_phi'] for r in selected]))
    print('FULL',json.dumps(data['fullft_comparisons'],indent=2))
    print('Exported',len(SOURCES),'hashed input files; all fixed-cohort checks passed.')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-training-only',action='store_true')
    parser.add_argument('--audit-eval-only',action='store_true')
    args=parser.parse_args()
    if args.audit_eval_only:
        audit_eval_traces()
    elif args.audit_training_only:
        audit_training_traces()
    else:
        main()
