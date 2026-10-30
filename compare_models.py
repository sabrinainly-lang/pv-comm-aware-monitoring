"""Algorithm comparison under the same week-blocked 5-fold protocol and final features.
Usage: python compare_models.py <clean_dataset.csv> <out.json>
Hyperparameters are fixed defaults, except 'HGB (tuned)', which is tuned only on the training folds (inner week-blocked CV)."""
import sys, json, time, itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import pvmon_core as C

d = C.load_clean(sys.argv[1]); y = d['Label'].values; F = C.build_features(d); fold = C.week_folds(d)
X = F[C.FEATURES].values
models = {
 'Logistic regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
 'k-NN (k=15)': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15, n_jobs=-1)),
 'MLP (64-32)': lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, random_state=42)),
 'Random Forest (100)': lambda: RandomForestClassifier(100, random_state=42, n_jobs=-1),
 'Extra Trees (100)': lambda: ExtraTreesClassifier(100, random_state=42, n_jobs=-1),
 'HGB (default)': lambda: HistGradientBoostingClassifier(random_state=42),
}
def inner_tune(Xtr, ytr, ftr):
    best, bs = None, -1
    for lr, leaf, it in itertools.product([0.05, 0.1], [20, 50], [200, 400]):
        s = []
        for k in sorted(set(ftr))[:3]:                      # 3 inner folds
            t = ftr != k
            m = HistGradientBoostingClassifier(learning_rate=lr, min_samples_leaf=leaf, max_iter=it, random_state=42).fit(Xtr[t], ytr[t])
            s.append((m.predict(Xtr[~t]) == ytr[~t]).mean())
        if np.mean(s) > bs: bs, best = np.mean(s), (lr, leaf, it)
    return best
R = {}; t0 = time.time()
for nm, mk in models.items():
    p = np.zeros(len(y), int); fa = []
    for k in range(C.N_FOLDS):
        tr = fold != k; p[~tr] = mk().fit(X[tr], y[tr]).predict(X[~tr]); fa.append(float((p[~tr] == y[~tr]).mean()))
    R[nm] = {'acc': float((p == y).mean()), 'macro_f1': float(f1_score(y, p, average='macro')), 'per_fold': fa}
    print(nm, R[nm]['acc'], round(time.time() - t0), flush=True)
p = np.zeros(len(y), int); hp = []; fa = []
for k in range(C.N_FOLDS):
    tr = fold != k; b = inner_tune(X[tr], y[tr], fold[tr]); hp.append(b)
    m = HistGradientBoostingClassifier(learning_rate=b[0], min_samples_leaf=b[1], max_iter=b[2], random_state=42).fit(X[tr], y[tr])
    p[~tr] = m.predict(X[~tr]); fa.append(float((p[~tr] == y[~tr]).mean()))
R['HGB (tuned on training folds only)'] = {'acc': float((p == y).mean()), 'macro_f1': float(f1_score(y, p, average='macro')), 'per_fold': fa, 'chosen_params': hp}
print('HGB tuned', R['HGB (tuned on training folds only)']['acc'], hp, flush=True)
json.dump(R, open(sys.argv[2], 'w'), indent=1)
