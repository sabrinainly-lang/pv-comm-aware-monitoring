"""Reproduces every number in the paper's results sections. Usage: python evaluate.py <clean_dataset.csv> <out.json> [n_seeds]"""
import sys, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import pvmon_core as C

CSV, OUT = sys.argv[1], sys.argv[2]; NSEED = int(sys.argv[3]) if len(sys.argv) > 3 else 20
t0 = time.time()
raw_all = pd.read_csv(CSV)
d = C.load_clean(CSV); y = d['Label'].values; F = C.build_features(d); fold = C.week_folds(d)
R = {'n_total_rows': int(len(raw_all)), 'n_clean': int(len(d)), 'n_comm_rows_removed': int(len(raw_all) - len(d))}

# --- 1. baseline = old dashboard pipeline (raw features, random 80/20 split) ---
Xa, Xb, ya, yb = train_test_split(F[C.RAW].values, y, test_size=.2, random_state=42, stratify=y)
m0 = C.make_model().fit(Xa, ya); p0 = m0.predict(Xb)
R['baseline_random_split'] = {'n_test': int(len(yb)), 'acc': float((p0 == yb).mean()), 'macro_f1': float(f1_score(yb, p0, average='macro'))}

# --- 2. week-blocked 5-fold CV ---
def cv_pred(mk, feats, proba=False):
    P = np.zeros((len(y), 4)) if proba else np.zeros(len(y), int)
    for k in range(C.N_FOLDS):
        tr = fold != k; m = mk().fit(F.loc[tr, feats].values, y[tr])
        P[~tr] = m.predict_proba(F.loc[~tr, feats].values) if proba else m.predict(F.loc[~tr, feats].values)
    return P
variants = {'raw': C.RAW, 'raw+ratios': C.RAW + C.RATIO, 'raw+ratios+time': C.RAW + C.RATIO + C.TIME, 'all (with 6-record window)': C.FEATURES}
R['cv_variants'] = {}
for nm, fe in variants.items():
    p = cv_pred(C.make_model, fe)
    R['cv_variants'][nm] = {'acc': float((p == y).mean()), 'macro_f1': float(f1_score(y, p, average='macro'))}
PR = cv_pred(C.make_model, C.FEATURES, proba=True); p = PR.argmax(1)
cm = confusion_matrix(y, p); pr, rc, f1, sup = precision_recall_fscore_support(y, p)
R['cv_rf'] = {'acc': float((p == y).mean()), 'macro_f1': float(f1.mean()), 'n': int(len(y)), 'confusion': cm.tolist(),
              'precision': pr.tolist(), 'recall': rc.tolist(), 'f1': f1.tolist(), 'support': sup.tolist(),
              'per_fold_acc': [float((p[fold == k] == y[fold == k]).mean()) for k in range(C.N_FOLDS)]}
ghi = d['GHI'].values; R['cv_rf']['by_ghi'] = {}
for lo, hi in [(0, 100), (100, 300), (300, 600), (600, 5000)]:
    mk = (ghi >= lo) & (ghi < hi); R['cv_rf']['by_ghi'][f'{lo}-{hi}'] = {'n': int(mk.sum()), 'acc': float((p[mk] == y[mk]).mean())}
# alarm confirmation (mean probability over last 12 valid records of the same day)
conf = np.zeros(len(y), int)
for lab in range(4):
    idx = np.flatnonzero(y == lab); cf = C.Confirmer(12)
    conf[idx] = [cf.update(ts, pr_) for ts, pr_ in zip(d['ts'].iloc[idx], PR[idx])]
R['cv_rf_confirmed12'] = {'acc': float((conf == y).mean()), 'macro_f1': float(f1_score(y, conf, average='macro'))}
# chronological stress test: train Jan-Oct, test Nov-Dec
tr, te = C.split_chronological(d, 10)
R['chrono'] = {}
for nm, fe in [('raw', C.RAW), ('all', C.FEATURES)]:
    m = C.make_model().fit(F.loc[tr, fe].values, y[tr]); R['chrono'][nm] = {'acc': float(m.score(F.loc[te, fe].values, y[te])), 'n_test': int(te.sum())}
print('classification done', round(time.time() - t0), 's', flush=True)

# --- 3. channel experiments on the held-out weeks (fold 0), model trained on the other weeks ---
trn = fold != C.HOLDOUT_FOLD
model = C.make_model().fit(F.loc[trn, C.FEATURES].values, y[trn])
H = C.holdout_stream_records(d); VH = H[C.RAW].values.astype(float); yh = H['Label'].values; tsh = H['ts']
R['holdout_stream'] = {'n': int(len(H))}

def run_stream(recv, crc_ok, vals, fill=None):
    """feature stream -> predictions. fill=None: gated (only accepted records); else unguarded with fill rule."""
    fz = C.Featurizer(); n = len(H); pred = -np.ones(n, int); last = VH[0].copy(); rows = []; idx = []
    for t in range(n):
        if fill is None:
            if not (recv[t] and crc_ok[t]): continue
            v = vals[t]
        else:
            if recv[t]: v = C.safe_values(vals[t]); last = v
            else: v = {'zero': np.zeros(7), 'sentinel': -np.ones(7), 'locf': last}[fill]
        rows.append(fz.update(tsh.iloc[t], dict(zip(C.RAW, v)))); idx.append(t)
    pred[idx] = model.predict(np.asarray(rows))
    return pred

R['channel'] = {}
for name in ['clean'] + C.SCENARIOS:
    L = []
    for sd in range(NSEED if name != 'clean' else 1):
        rng = np.random.default_rng(sd)
        truth, recv, vals, crc_ok = C.impair(name, VH, rng)
        mon = C.ChannelMonitor(); lab = np.array([mon.update(t, recv[t], crc_ok[t]) for t in range(len(H))])
        imp = truth > 0; acc_mask = lab == 0
        pg = run_stream(recv, crc_ok, vals)
        r = {'impaired_frac': float(imp.mean()),
             'det_rate': float((lab[imp] > 0).mean()) if imp.any() else None,
             'type_acc': float((lab[imp] == truth[imp]).mean()) if imp.any() else None,
             'false_alarm_ok': float((lab[~imp] > 0).mean()),
             'gated_coverage': float(acc_mask.mean()), 'gated_acc_accepted': float((pg[acc_mask] == yh[acc_mask]).mean()),
             'gated_wrong_on_impaired': float(((pg[imp] >= 0)).mean()) if imp.any() else 0.0}
        for fill in ['zero', 'sentinel', 'locf']:
            pn = run_stream(recv, crc_ok, vals, fill)
            r[f'naive_{fill}_acc_imp'] = float((pn[imp] == yh[imp]).mean()) if imp.any() else None
            r[f'naive_{fill}_acc_all'] = float((pn == yh).mean())
            if fill in ('zero', 'sentinel') and imp.any():
                r[f'naive_{fill}_pred_dist'] = np.bincount(pn[imp], minlength=4).tolist()
        if name == 'noise20':
            pf = ~np.isfinite(vals).all(1)
            with np.errstate(invalid='ignore'):
                lo_, hi_ = VH[:0].min() if False else np.nanmin(d[C.RAW].values, 0), np.nanmax(d[C.RAW].values, 0); span = hi_ - lo_
                pf |= ((vals < lo_ - .5 * span) | (vals > hi_ + .5 * span)).any(1)
            r['plaus_det'] = float(pf[imp].mean()); r['plaus_fa'] = float(pf[~imp].mean())
        L.append(r)
    df = pd.DataFrame(L)
    R['channel'][name] = {'mean': {k: (None if df[k].isna().all() else float(np.nanmean(df[k]))) for k in df.columns if not k.endswith('pred_dist')},
                          'std': {k: (None if df[k].isna().all() else float(np.nanstd(df[k], ddof=1)) if len(df) > 1 else 0.0) for k in df.columns if not k.endswith('pred_dist')},
                          'pred_dist_zero': np.mean([r_['naive_zero_pred_dist'] for r_ in L if 'naive_zero_pred_dist' in r_], axis=0).tolist() if name != 'clean' else None,
                          'pred_dist_sentinel': np.mean([r_['naive_sentinel_pred_dist'] for r_ in L if 'naive_sentinel_pred_dist' in r_], axis=0).tolist() if name != 'clean' else None}
    print(name, 'done', round(time.time() - t0), 's', flush=True)
R['meta'] = {'n_seeds': NSEED, 'outage_threshold': C.OUTAGE_TH, 'window': C.WINDOW, 'features': C.FEATURES}
json.dump(R, open(OUT, 'w'), indent=1)
print(json.dumps({k: R[k] for k in ['baseline_random_split', 'cv_variants', 'chrono', 'cv_rf_confirmed12']}, indent=1))
print('cv_rf acc', R['cv_rf']['acc'], R['cv_rf']['per_fold_acc'])
