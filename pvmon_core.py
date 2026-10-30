"""Core logic shared by the dashboard (app.py), the evaluation script and the stream generator.
Keeps the paper's numbers and the dashboard's numbers on exactly the same code path."""
import zlib
from collections import deque
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore', category=Warning, module='sklearn')

# ---------------- dataset ----------------
SRC_COLS = {
    'ts': 'Time stamp',
    'P': 'System power generated | (kW)',
    'GHI': 'Irradiance GHI from weather file | (W/m2)',
    'AC': 'Inverter AC output power | (kW)',
    'T': 'Weather file ambient temperature | (C)',
    'DC': 'Inverter DC input power | (kW)',
    'eff': 'Inverter efficiency | (%)',
    'clp': 'Inverter clipping loss DC MPPT voltage limits | (kW)',
}
RAW = ['P', 'GHI', 'AC', 'T', 'DC', 'eff', 'clp']
RATIO = ['DC_GHI', 'AC_GHI', 'AC_DC', 'clp_DC']
TIME = ['hour', 'decl']
ROLL = ['DC_GHI_r6', 'AC_GHI_r6', 'AC_DC_r6']
FEATURES = RAW + RATIO + TIME + ROLL
WINDOW = 6                      # past valid records, same calendar day
CLASS_NAMES = {0: 'Normal', 1: 'Soiling Fault', 2: 'DC Wiring Fault', 3: 'Inverter Fault'}
CHANNEL_NAMES = {0: 'OK', 1: 'Packet loss', 2: 'Noise', 3: 'Reduced sampling', 4: 'Outage'}
OUTAGE_TH = 5                   # consecutive missing slots -> outage


def parse_ts(s):
    """'Jan 1, 06:30 am' -> Timestamp (year 2019, the calendar year of the NSRDB weather file)."""
    return pd.to_datetime('2019 ' + s.astype(str), format='%Y %b %d, %I:%M %p')


def load_clean(path):
    """Load the SAM CSV; keep only records received correctly (Communication_Error == 0)."""
    d = pd.read_csv(path)
    inv = {v: k for k, v in SRC_COLS.items()}
    d = d.rename(columns=inv)
    d['ts'] = parse_ts(d['ts'])
    if 'Communication_Error' not in d.columns:
        d['Communication_Error'] = 0
    d = d[d['Communication_Error'] == 0].reset_index(drop=True)
    return d[['ts'] + RAW + ['Label']]


# ---------------- features (identical offline and streaming) ----------------
class Featurizer:
    """Streaming feature builder: ratios of the current record + mean of the last WINDOW valid
    records of the same calendar day. Time features come from the receiver clock."""
    def __init__(self):
        self.day = None
        self.hist = {k: deque(maxlen=WINDOW) for k in ('DC_GHI', 'AC_GHI', 'AC_DC')}

    def update(self, ts, v):
        day = (ts.month, ts.day)
        if day != self.day:
            self.day = day
            for q in self.hist.values(): q.clear()
        g = max(float(v['GHI']), 1.0); dc = max(float(v['DC']), 1e-3)
        r = {'DC_GHI': v['DC'] / g, 'AC_GHI': v['AC'] / g, 'AC_DC': v['AC'] / dc}
        for k, q in self.hist.items(): q.append(r[k])
        f = [v[c] for c in RAW] + [r['DC_GHI'], r['AC_GHI'], r['AC_DC'], v['clp'] / dc,
             ts.hour + ts.minute / 60.0, 23.45 * np.sin(2 * np.pi * (284 + ts.dayofyear) / 365.0)] + [float(np.mean(self.hist[k])) for k in ('DC_GHI', 'AC_GHI', 'AC_DC')]
        return f


def build_features(df):
    """Offline features for a frame of valid records: one stream per Label block (file order = time order)."""
    out = []
    for lab, g in df.groupby('Label', sort=True):
        fz = Featurizer()
        rows = [fz.update(ts, r) for ts, r in zip(g['ts'], g[RAW].to_dict('records'))]
        out.append(pd.DataFrame(rows, columns=FEATURES, index=g.index))
    return pd.concat(out).loc[df.index]


def split_chronological(df, last_train_month=10):
    tr = (df['ts'].dt.month <= last_train_month).values
    return tr, ~tr


def make_rf(n_estimators=100, seed=42):
    """Random Forest (the reference model of the first version of the paper)."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)


def make_model(seed=42):
    """Final classifier: standardized inputs + MLP with two hidden layers (64, 32)."""
    return make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, random_state=seed))


def train(df, feats=FEATURES, last_train_month=10):
    F = build_features(df)
    tr, te = split_chronological(df, last_train_month)
    m = make_model().fit(F.loc[tr, feats].values, df.loc[tr, 'Label'].values)
    return m, F, tr, te


# ---------------- channel: receiver-side monitor ----------------
class ChannelMonitor:
    """Causal receiver rules (Table V of the paper). One call per time slot."""
    def __init__(self, th=OUTAGE_TH):
        self.th = th; self.miss = 0; self.arr = []

    def update(self, slot, received, crc_ok=True):
        if received:
            self.miss = 0; self.arr.append(slot)
            return 0 if crc_ok else 2
        self.miss += 1
        ia = np.diff(self.arr[-4:]) if len(self.arr) >= 4 else np.array([])
        if len(ia) == 3 and ia[0] == ia[1] == ia[2] and ia[0] >= 2 and self.miss < ia[0]:
            return 3
        return 4 if self.miss >= self.th else 1


def frame_crc(values):
    return zlib.crc32(np.asarray(values, dtype=np.float64).tobytes())


# ---------------- channel: impairment generator ----------------
def gilbert(rng, n, p_gb, p_bg):
    s = np.zeros(n, bool); bad = False
    for t in range(n):
        bad = (rng.random() < p_gb) if not bad else not (rng.random() < p_bg)
        s[t] = bad
    return s


def corrupt_row(rng, row):
    b = bytearray(np.asarray(row, dtype=np.float64).tobytes())
    for _ in range(rng.integers(1, 4)):
        k = rng.integers(0, len(b) * 8); b[k // 8] ^= 1 << (k % 8)
    return np.frombuffer(bytes(b), dtype=np.float64).copy()


SCENARIOS = ['loss10', 'loss30', 'noise20', 'down2', 'down5', 'outage20']


def impair(name, V, rng):
    """V: (n,7) true payloads. Returns truth(int codes), received(bool), payload(as received), crc_ok(bool)."""
    n = len(V); truth = np.zeros(n, int); recv = np.ones(n, bool); vals = V.copy(); crc_ok = np.ones(n, bool)
    if name.startswith('loss'):
        p = {'loss10': .0556, 'loss30': .2143}[name]
        s = gilbert(rng, n, p, .5); recv[s] = False; truth[s] = 1
    elif name == 'noise20':
        s = rng.random(n) < .2
        for t in np.flatnonzero(s):
            vals[t] = corrupt_row(rng, V[t])
            if frame_crc(vals[t]) != frame_crc(V[t]):
                truth[t] = 2; crc_ok[t] = False
    elif name.startswith('down'):
        k = int(name[4:]); off = rng.integers(0, k)
        s = (np.arange(n) % k) != off; recv[s] = False; truth[s] = 3
    elif name == 'outage20':
        s = np.zeros(n, bool); L = 20
        for st in rng.choice(np.arange(0, n - L, 2 * L), size=int(.1 * n / L), replace=False): s[st:st + L] = True
        recv[s] = False; truth[s] = 4
    elif name == 'mixed':
        seg = n // 8; plan = [None, 'loss10', None, 'noise20', None, 'down5', None, 'outage20']
        for i, nm in enumerate(plan):
            a, b = i * seg, (i + 1) * seg if i < 7 else n
            if nm is None: continue
            t_, r_, v_, c_ = impair(nm, V[a:b], rng)
            truth[a:b] = t_; recv[a:b] = r_; vals[a:b] = v_; crc_ok[a:b] = c_
    elif name != 'clean':
        raise ValueError(name)
    return truth, recv, vals, crc_ok


def safe_values(v):
    return np.clip(np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0., posinf=1e30, neginf=-1e30), -1e30, 1e30)


# ---------------- evaluation folds ----------------
N_FOLDS = 5
HOLDOUT_FOLD = 0                # weeks used for the replay stream and the dashboard demo


def week_folds(df, k=N_FOLDS):
    """Whole 7-day blocks are held out together (no day or week is split between train and test)."""
    return (((df['ts'].dt.dayofyear - 1) // 7) % k).values


# ---------------- streaming processor (used by the dashboard, the evaluation and the CSV test) ----------------
class Confirmer:
    """Alarm confirmation: mean class probability over the last N valid records of the same day."""
    def __init__(self, n=12):
        self.q = deque(maxlen=n); self.day = None

    def update(self, ts, proba):
        day = (ts.month, ts.day)
        if day != self.day:
            self.day = day; self.q.clear()
        self.q.append(proba)
        return int(np.argmax(np.mean(self.q, axis=0)))


class StreamProcessor:
    """One call per time slot: channel gate first, electrical classifier only for accepted records."""
    def __init__(self, model):
        getattr(model, 'set_params', lambda **k: None)(**({'n_jobs': 1} if hasattr(model, 'n_jobs') else {}))          # single-row predictions: thread start-up would dominate
        self.model = model; self.mon = ChannelMonitor(); self.fz = Featurizer(); self.conf = Confirmer()

    def step(self, slot, ts, received, crc_ok, values):
        ch = self.mon.update(slot, bool(received), bool(crc_ok))
        out = {'channel': ch, 'pred': None, 'confirmed': None}
        if ch == 0:
            v = dict(zip(RAW, np.asarray(values, dtype=float)))
            f = self.fz.update(ts, v)
            pr = self.model.predict_proba(np.asarray(f).reshape(1, -1))[0]
            out['pred'] = int(self.model.classes_[int(np.argmax(pr))])
            out['confirmed'] = int(self.model.classes_[self.conf.update(ts, pr)])
        return out


# ---------------- stream CSV (the file the dashboard replays) ----------------
def make_stream_frame(df_valid, scenario, seed):
    """df_valid: valid records (ts, RAW, Label) in file order. Returns the CSV frame."""
    rng = np.random.default_rng(seed)
    V = df_valid[RAW].values.astype(np.float64)
    truth, recv, vals, crc_ok = impair(scenario, V, rng)
    out = pd.DataFrame({'scenario': scenario, 'slot': np.arange(len(V)),
                        'Time stamp': df_valid['ts'].dt.strftime('%b %-d, %I:%M %p').str.lower().values,
                        'received': recv.astype(int), 'crc_ok': np.where(recv, crc_ok.astype(int), 0)})
    for j, k in enumerate(RAW):
        col = vals[:, j].astype(object); col[~recv] = np.nan
        out[SRC_COLS[k]] = col.astype(float)
    out['truth_Label'] = df_valid['Label'].values
    out['truth_channel'] = truth
    return out


def holdout_stream_records(df):
    f = week_folds(df)
    return df[f == HOLDOUT_FOLD].sort_values(['Label', 'ts'], kind='stable').reset_index(drop=True)
