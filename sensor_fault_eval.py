"""Challenge test for the gate: invalid sensor values that arrive in a valid frame (checksum computed over the faulty payload).
The receiver gate accepts such records by construction; the test measures what the classifier then outputs.
Usage: python sensor_fault_eval.py <clean_dataset.csv> <out.json>"""
import sys, json
import numpy as np
import pvmon_core as C
CSV, OUT = sys.argv[1], sys.argv[2]
d = C.load_clean(CSV); y = d['Label'].values; F = C.build_features(d); fold = C.week_folds(d)
model = C.make_model().fit(F.loc[fold != C.HOLDOUT_FOLD, C.FEATURES].values, y[fold != C.HOLDOUT_FOLD])
H = C.holdout_stream_records(d); V = H[C.RAW].values.astype(float); yh = H['Label'].values; ts = H['ts']
KINDS = {'GHI reads 0 (irradiance sensor offline)': lambda v: v.__setitem__(C.RAW.index('GHI'), 0.0),
         'all seven values -1 with a valid checksum': lambda v: v.__setitem__(slice(None), -1.0),
         'temperature reads 100 C (stuck sensor)': lambda v: v.__setitem__(C.RAW.index('T'), 100.0)}
R = {}
for name, fn in KINDS.items():
    out = []
    for seed in range(5):
        rng = np.random.default_rng(seed); bad = rng.random(len(H)) < 0.10
        fz = C.Featurizer(); rows = []
        for t in range(len(H)):
            v = V[t].copy()
            if bad[t]: fn(v)
            crc_ok = C.frame_crc(v) == C.frame_crc(v)            # checksum computed over the transmitted (faulty) payload
            assert crc_ok
            rows.append(fz.update(ts.iloc[t], dict(zip(C.RAW, v))))
        p = model.predict(np.asarray(rows))
        out.append({'gate_accepts': 1.0, 'acc_on_bad': float((p[bad] == yh[bad]).mean()), 'acc_on_good': float((p[~bad] == yh[~bad]).mean()),
                    'pred_dist_bad': np.bincount(p[bad], minlength=4) / bad.sum()})
    R[name] = {'acc_on_bad': float(np.mean([o['acc_on_bad'] for o in out])), 'acc_on_unaffected': float(np.mean([o['acc_on_good'] for o in out])),
               'pred_dist_bad': np.mean([o['pred_dist_bad'] for o in out], axis=0).tolist(), 'gate_accepts': 1.0}
    print(name, R[name], flush=True)
json.dump(R, open(OUT, 'w'), indent=1)
