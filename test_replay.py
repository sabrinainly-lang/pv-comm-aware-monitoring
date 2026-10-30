"""Headless test of the released stream CSV: replays every scenario through the same StreamProcessor the dashboard uses
and scores it against the truth columns. Usage: python test_replay.py <training.csv> <stream.csv> [out.json]"""
import sys, json
import numpy as np, pandas as pd
import pvmon_core as C

tr_csv, st_csv = sys.argv[1], sys.argv[2]; out = sys.argv[3] if len(sys.argv) > 3 else 'replay_results.json'
d = C.load_clean(tr_csv); F = C.build_features(d); y = d['Label'].values; fold = C.week_folds(d)
model = C.make_model().fit(F.loc[fold != C.HOLDOUT_FOLD, C.FEATURES].values, y[fold != C.HOLDOUT_FOLD])
S = pd.read_csv(st_csv); res = {}
for sc, g in S.groupby('scenario', sort=False):
    g = g.reset_index(drop=True); proc = C.StreamProcessor(model)
    ts = C.parse_ts(g['Time stamp']); cols = [C.SRC_COLS[k] for k in C.RAW]
    ch = np.zeros(len(g), int); pred = -np.ones(len(g), int); conf = -np.ones(len(g), int)
    for i in range(len(g)):
        r = g.iloc[i]; ok = int(r['received']) and int(r['crc_ok'])
        o = proc.step(int(r['slot']), ts.iloc[i], int(r['received']), int(r['crc_ok']), r[cols].values.astype(float) if ok else None)
        ch[i] = o['channel']
        if o['pred'] is not None: pred[i] = o['pred']; conf[i] = o['confirmed']
    tc = g['truth_channel'].values; tl = g['truth_Label'].values; imp = tc > 0; acc = ch == 0
    res[sc] = {'slots': int(len(g)), 'impaired_frac': float(imp.mean()),
               'detection_rate': float((ch[imp] > 0).mean()) if imp.any() else None,
               'type_accuracy': float((ch[imp] == tc[imp]).mean()) if imp.any() else None,
               'false_channel_alarms': float((ch[~imp] > 0).mean()),
               'classified_share': float(acc.mean()),
               'electrical_acc_on_accepted': float((pred[acc] == tl[acc]).mean()),
               'confirmed_acc_on_accepted': float((conf[acc] == tl[acc]).mean()),
               'electrical_labels_on_impaired_slots': int((pred[imp] >= 0).sum())}
json.dump(res, open(out, 'w'), indent=1)
print(pd.DataFrame(res).T.round(3).to_string())
