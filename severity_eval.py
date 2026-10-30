"""Severity-sweep and independent-weather evaluation (reviewer notes 1, 2 and 7).
Usage: python severity_eval.py manifest.csv out.json
manifest.csv columns: file, label (0 normal, 1 soiling, 2 DC wiring, 3 inverter), severity (e.g. soiling loss in %, any number),
                      weather (w1 = the original weather file, w2 = an independent year or site), role (train | test | weeks)  -- 'weeks' = week-blocked split inside the file (fold 0 = test), for the normal file
Each file is a SAM export with the same columns as the project CSV (Time stamp, seven variables, no communication flag needed).
Rules enforced here: the scaler and network are fitted on role=train rows only; test rows are never seen in training;
the history window is built inside each file (one stream per file) and only from past records of the same day."""
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import pvmon_core as C

man = pd.read_csv(sys.argv[1]); out = sys.argv[2]
parts = []
for _, r in man.iterrows():
    d = pd.read_csv(r['file']).rename(columns={v: k for k, v in C.SRC_COLS.items()})
    d['ts'] = C.parse_ts(d['ts'])
    if 'Communication_Error' in d.columns: d = d[d['Communication_Error'] == 0]
    d = d[C.RAW + ['ts']].copy(); d['Label'] = int(r['label'])
    d = d[d['GHI'] > 0].reset_index(drop=True)                      # same daylight rule as the main dataset
    F = C.build_features(d)
    F['y'] = d['Label'].values; F['severity'] = r['severity']; F['weather'] = r['weather']; F['role'] = r['role']
    F['role'] = np.where(C.week_folds(d) == C.HOLDOUT_FOLD, 'test', 'train') if r['role'] == 'weeks' else r['role']
    F['file'] = r['file']; F['dc_ghi_hi'] = np.where(d['GHI'] > 400, d['DC'] / d['GHI'].clip(lower=1), np.nan)
    parts.append(F)
A = pd.concat(parts, ignore_index=True)
tr = A[A['role'] == 'train']; te = A[A['role'] == 'test']
assert len(tr) and len(te), 'manifest needs both train and test rows'
model = C.make_model().fit(tr[C.FEATURES].values, tr['y'].values)
A.loc[te.index, 'pred'] = model.predict(te[C.FEATURES].values)
te = A.loc[te.index]

R = {'n_train': int(len(tr)), 'n_test': int(len(te)), 'train_severities': sorted(map(str, tr.groupby(['y', 'severity']).size().index.tolist()))}
R['accuracy_unseen'] = float((te['pred'] == te['y']).mean())
R['macro_f1_unseen'] = float(f1_score(te['y'], te['pred'].astype(int), average='macro'))
R['confusion'] = confusion_matrix(te['y'], te['pred'].astype(int), labels=[0, 1, 2, 3]).tolist()
rows = []
base = A[(A['y'] == 0) & (A['weather'] == 'w1')]['dc_ghi_hi'].mean()
for (y, sev, wx, role), g in A.groupby(['y', 'severity', 'weather', 'role']):
    row = {'label': int(y), 'severity': sev, 'weather': wx, 'role': role, 'n': int(len(g)),
           'dc_per_ghi_vs_normal_pct': float(100 * (g['dc_ghi_hi'].mean() / base - 1))}
    if role == 'test':
        t = te.loc[g.index]; row['recall'] = float((t['pred'] == t['y']).mean())
        row['pred_dist'] = np.bincount(t['pred'].astype(int), minlength=4).tolist()
    rows.append(row)
R['per_scenario'] = rows
json.dump(R, open(out, 'w'), indent=1)
print(pd.DataFrame(rows).to_string(index=False)); print('accuracy on unseen severities / weather:', round(R['accuracy_unseen'], 4))
