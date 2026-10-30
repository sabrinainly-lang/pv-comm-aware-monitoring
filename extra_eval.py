"""Additional analyses requested in the peer review (uncertainty, ablations, transition tests).
Usage: python extra_eval.py <clean_dataset.csv> <out.json>
1. week-cluster bootstrap 95% intervals for the week-blocked cross-validation result;
2. ablations (time-only, no time features, no clipping variables, scale-free inputs);
3. causal transition tests on the held-out weeks: fault onset, recovery, onset during an outage, false alarms."""
import sys, json, time
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
import pvmon_core as C

CSV, OUT = sys.argv[1], sys.argv[2]; t0 = time.time()
d = C.load_clean(CSV); y = d['Label'].values; F = C.build_features(d); fold = C.week_folds(d)
wk = ((d['ts'].dt.dayofyear - 1) // 7).values
SKIP_CV = len(sys.argv) > 3 and sys.argv[3] == 'transitions'          # reuse the cross-validation part of an existing output file
R = json.load(open(OUT)) if SKIP_CV else {'n': int(len(d)), 'n_weeks': int(len(np.unique(wk)))}

def cv_proba(feats):
    P = np.zeros((len(y), 4))
    for k in range(C.N_FOLDS):
        tr = fold != k; m = C.make_model().fit(F.loc[tr, feats].values, y[tr]); P[~tr] = m.predict_proba(F.loc[~tr, feats].values)
    return P

def boot(p, B=2000, seed=0):
    weeks = np.unique(wk); cms = np.array([confusion_matrix(y[wk == w], p[wk == w], labels=[0, 1, 2, 3]) for w in weeks])
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        cm = cms[rng.integers(0, len(weeks), len(weeks))].sum(0)
        rec = np.diag(cm) / np.maximum(cm.sum(1), 1); prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9); out.append([np.trace(cm) / cm.sum(), f1.mean(), *rec])
    out = np.array(out); lo, hi = np.percentile(out, [2.5, 97.5], axis=0)
    cm = confusion_matrix(y, p, labels=[0, 1, 2, 3]); rec = np.diag(cm) / cm.sum(1); prec = np.diag(cm) / cm.sum(0)
    pt = [np.trace(cm) / cm.sum(), float((2 * prec * rec / (prec + rec)).mean()), *rec]
    names = ['acc', 'macro_f1', 'recall_normal', 'recall_soiling', 'recall_wiring', 'recall_inverter']
    return {n: {'est': float(pt[i]), 'lo': float(lo[i]), 'hi': float(hi[i])} for i, n in enumerate(names)}

# ---- 1 and 2: intervals and ablations ----
V = {'all 16 features': C.FEATURES,
     'time only (hour, declination)': C.TIME,
     'no time features (raw + ratios + window)': C.RAW + C.RATIO + C.ROLL,
     'no clipping variables (14 features)': [f for f in C.FEATURES if f not in ('clp', 'clp_DC')],
     'scale-free only (ratios + time + window)': C.RATIO + C.TIME + C.ROLL,
     'raw seven variables': C.RAW}
if not SKIP_CV:
    R['cv'] = {}; PROB = {}
    for nm, fe in V.items():
        PROB[nm] = cv_proba(fe); R['cv'][nm] = boot(PROB[nm].argmax(1)); print(nm, round(R['cv'][nm]['acc']['est'], 4), round(time.time() - t0), 's', flush=True)
    R['per_fold_sd_all'] = float(np.std([ (PROB['all 16 features'].argmax(1)[fold == k] == y[fold == k]).mean() for k in range(C.N_FOLDS)], ddof=1))

# ---- 3: causal transition tests ----
trn = fold != C.HOLDOUT_FOLD
model = C.make_model().fit(F.loc[trn, C.FEATURES].values, y[trn])
H = d[fold == C.HOLDOUT_FOLD].copy(); H['day'] = list(zip(H['ts'].dt.month, H['ts'].dt.day))
rec = {lab: g.set_index('ts') for lab, g in H.groupby('Label')}
days = sorted(set(H['day'])); NAME = {0: 'Normal', 1: 'Soiling', 2: 'DC wiring', 3: 'Inverter'}

def run_day(seq, outage=None, flush=False):
    """seq: list of (ts, raw-dict) in time order. outage=(start, length) in slot index. Returns confirmed and instantaneous predictions (None where no output)."""
    fz = C.Featurizer(); cf = C.Confirmer(12); mon = C.ChannelMonitor(); n = len(seq)
    conf = [None] * n; inst = [None] * n; ch = np.zeros(n, int)
    for t, (ts, v) in enumerate(seq):
        miss = outage is not None and outage[0] <= t < outage[0] + outage[1]
        c = mon.update(t, not miss, True); ch[t] = c
        if c == 4 and flush and (t == outage[0] + C.OUTAGE_TH - 1): fz = C.Featurizer(); cf = C.Confirmer(12)
        if miss: continue
        f = fz.update(ts, v); pr = model.predict_proba(np.asarray(f).reshape(1, -1))[0]
        inst[t] = int(model.classes_[int(np.argmax(pr))]); conf[t] = int(model.classes_[cf.update(ts, pr)])
    return conf, inst, ch

def stable_delay(out, k, target):
    """records (counted from slot k) until the output equals target and stays equal to the end of the day; None if never."""
    n = len(out); ok = [o is None or o == target for o in out]
    for t in range(k, n):
        if out[t] == target and all(ok[t:]): return sum(o is not None for o in out[k:t])
    return None
def first_delay(out, k, target):
    for t in range(k, len(out)):
        if out[t] == target: return sum(o is not None for o in out[k:t])
    return None

def build(day, A, B, k):
    ts_common = sorted(set.intersection(*[set(rec[l].index[[x == day for x in rec[l]['day']]]) for l in range(4)]))
    seq = []
    for t, ts in enumerate(ts_common):
        row = rec[A if t < k else B].loc[ts]; seq.append((ts, {c: float(row[c]) for c in C.RAW}))
    return seq

def summarize(x):
    v = [a for a in x if a is not None]
    return {'n': len(x), 'frac_within_day': len(v) / max(len(x), 1), 'median_records': float(np.median(v)) if v else None,
            'p90_records': float(np.percentile(v, 90)) if v else None}

T = {'onset': {}, 'outage': {}}; RAW_L = {}; RAW_O = {}
for A in range(4):
    for B in range(4):
        if A == B: continue
        L = {'inst_first': [], 'inst_stable': [], 'conf_first': [], 'conf_stable': []}
        O = {pol: {'conf_stable': [], 'conf_first': []} for pol in ('persist', 'flush')}
        for day in days:
            n = len(set.intersection(*[set(rec[l].index[[x == day for x in rec[l]['day']]]) for l in range(4)]))
            if n < 16: continue
            k = n // 2; seq = build(day, A, B, k); conf, inst, _ = run_day(seq)
            L['inst_first'].append(first_delay(inst, k, B)); L['inst_stable'].append(stable_delay(inst, k, B))
            L['conf_first'].append(first_delay(conf, k, B)); L['conf_stable'].append(stable_delay(conf, k, B))
            os_ = k - 2; recon = os_ + 10
            if recon + 4 < n:
                for pol, fl in (('persist', False), ('flush', True)):
                    cf2, _, _ = run_day(seq, outage=(os_, 10), flush=fl)
                    O[pol]['conf_stable'].append(stable_delay(cf2, recon, B)); O[pol]['conf_first'].append(first_delay(cf2, recon, B))
        T['onset'][f'{NAME[A]}->{NAME[B]}'] = {m: summarize(v) for m, v in L.items()}
        T['outage'][f'{NAME[A]}->{NAME[B]}'] = {pol: {m: summarize(v) for m, v in O[pol].items()} for pol in O}
        RAW_L[f'{NAME[A]}->{NAME[B]}'] = L; RAW_O[f'{NAME[A]}->{NAME[B]}'] = O
    print('transitions from', NAME[A], round(time.time() - t0), 's', flush=True)

# pooled transition summaries (fault onset: Normal -> fault; recovery: fault -> Normal; fault-to-fault), exact pooling of all delays
def pooled(keys, src, metric, pol=None):
    xs = []
    for k in keys:
        xs += (src[k][pol][metric] if pol else src[k][metric])
    v = [a for a in xs if a is not None]
    return {'n': len(xs), 'frac_within_day': len(v) / len(xs), 'median_records': float(np.median(v)) if v else None,
            'p90_records': float(np.percentile(v, 90)) if v else None}
onset = [f'Normal->{NAME[b]}' for b in (1, 2, 3)]; recov = [f'{NAME[a]}->Normal' for a in (1, 2, 3)]; f2f = [f'{NAME[a]}->{NAME[b]}' for a in (1, 2, 3) for b in (1, 2, 3) if a != b]
GR = (('onset_from_normal', onset), ('recovery_to_normal', recov), ('fault_to_fault', f2f))
T['pooled'] = {g: {m: pooled(ks, RAW_L, m) for m in ('inst_first', 'inst_stable', 'conf_first', 'conf_stable')} for g, ks in GR}
T['pooled_outage'] = {g: {pol: {m: pooled(ks, RAW_O, m, pol) for m in ('conf_first', 'conf_stable')} for pol in ('persist', 'flush')} for g, ks in GR}

# false alarms on pure Normal days (held-out weeks)
fa_days = 0; fa_events = 0; fa_rec = 0; nrec = 0; ndays = 0
for day in days:
    n = len(set.intersection(*[set(rec[l].index[[x == day for x in rec[l]['day']]]) for l in range(4)]))
    if n < 16: continue
    seq = build(day, 0, 0, n); conf, inst, _ = run_day(seq); ndays += 1
    al = np.array([c != 0 for c in conf]); fa_rec += int(al.sum()); nrec += n
    ev = int(((al[1:] & ~al[:-1]).sum()) + al[0]); fa_events += ev; fa_days += int(ev > 0)
T['false_alarms_normal_days'] = {'days': ndays, 'days_with_alarm': fa_days, 'alarm_events': fa_events, 'events_per_day': fa_events / ndays, 'records': nrec, 'alarm_record_rate': fa_rec / nrec}
R['transitions'] = T; R['transition_setup'] = {'days_in_holdout': len(days), 'records_per_day_median': float(np.median([len(set.intersection(*[set(rec[l].index[[x == day for x in rec[l]['day']]]) for l in range(4)])) for day in days]))}
json.dump(R, open(OUT, 'w'), indent=1); print('done', round(time.time() - t0), 's')
