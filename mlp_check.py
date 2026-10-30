"""Robustness of the MLP result: 5 seeds (week-blocked CV), chronological split, confusion matrix.
Usage: python mlp_check.py <clean_dataset.csv> <out.json>"""
import sys, json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import pvmon_core as C
d = C.load_clean(sys.argv[1]); y = d['Label'].values; F = C.build_features(d); fold = C.week_folds(d); X = F[C.FEATURES].values
mk = lambda s: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, random_state=s))
R = {'seeds': {}}
for s in range(5):
    p = np.zeros(len(y), int)
    for k in range(C.N_FOLDS):
        tr = fold != k; p[~tr] = mk(s).fit(X[tr], y[tr]).predict(X[~tr])
    R['seeds'][s] = {'acc': float((p == y).mean()), 'macro_f1': float(f1_score(y, p, average='macro'))}
    if s == 0: R['confusion_seed0'] = confusion_matrix(y, p).tolist()
    print('seed', s, R['seeds'][s], flush=True)
a = [v['acc'] for v in R['seeds'].values()]; R['mean_acc'] = float(np.mean(a)); R['std_acc'] = float(np.std(a))
tr, _ = C.split_chronological(d)
for nm, cols in [('final', C.FEATURES), ('raw', C.RAW)]:
    m = mk(0).fit(F.loc[tr, cols].values, y[tr]); pc = m.predict(F.loc[~tr, cols].values)
    R['chrono_' + nm] = float((pc == y[~tr]).mean()); print('chrono', nm, R['chrono_' + nm], flush=True)
json.dump(R, open(sys.argv[2], 'w'), indent=1)
