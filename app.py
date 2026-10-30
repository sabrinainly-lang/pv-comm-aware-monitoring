import time
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st

import pvmon_core as C

st.set_page_config(page_title="Solar PV Monitoring Dashboard", page_icon="☀️", layout="wide")
st.title("☀️ Solar PV Monitoring Dashboard")
st.caption("Channel check first, electrical fault classification second (neural network)")

TRAIN_CSV = "Solar_Fault_Dataset_With_Comm_Loss_No_New_Label.csv"
STREAM_CSV = "Solar_PV_Comm_Stream_Test.csv"

# ------------------------------------------------------------------ model (cached)
@st.cache_resource(show_spinner="Training the neural network on the SAM dataset ...")
def get_model(path):
    d = C.load_clean(path)
    F = C.build_features(d)
    y = d["Label"].values
    fold = C.week_folds(d)
    train = fold != C.HOLDOUT_FOLD                       # weeks used by the replay stream are never trained on
    model = C.make_model().fit(F.loc[train, C.FEATURES].values, y[train])
    proba = model.predict_proba(F.loc[~train, C.FEATURES].values)
    pred = model.classes_[proba.argmax(1)]
    acc = float((pred == y[~train]).mean())
    conf = np.zeros(int((~train).sum()), int)
    h = d.loc[~train]
    k = 0
    for lab in sorted(h["Label"].unique()):               # confirmation is evaluated stream by stream
        idx = np.flatnonzero(h["Label"].values == lab)
        cf = C.Confirmer(12)
        conf[idx] = [model.classes_[cf.update(ts, p)] for ts, p in zip(h["ts"].iloc[idx], proba[idx])]
    acc_conf = float((conf == h["Label"].values).mean())
    return model, acc, acc_conf, int((~train).sum())


@st.cache_data(show_spinner=False)
def get_stream(path_or_bytes):
    return pd.read_csv(path_or_bytes)


up_train = st.file_uploader("Training dataset (SAM CSV) - optional", type=["csv"])
up_stream = st.file_uploader("Communication stream CSV - optional", type=["csv"])
try:
    model, acc, acc_conf, n_hold = get_model(up_train if up_train is not None else TRAIN_CSV)
except FileNotFoundError:
    st.warning("Training dataset not found. Upload the SAM CSV.")
    st.stop()
try:
    stream = get_stream(up_stream if up_stream is not None else STREAM_CSV)
except FileNotFoundError:
    st.warning("Stream CSV not found. Run generate_comm_csv.py or upload a stream file.")
    st.stop()

need = ["scenario", "slot", "Time stamp", "received", "crc_ok"] + [C.SRC_COLS[k] for k in C.RAW]
missing = [c for c in need if c not in stream.columns]
if missing:
    st.error("Stream CSV is missing columns: " + ", ".join(missing))
    st.stop()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Dashboard Settings")
scenario = st.sidebar.selectbox("Channel scenario", list(stream["scenario"].unique()),
                                index=list(stream["scenario"].unique()).index("mixed") if "mixed" in set(stream["scenario"]) else 0)
sub = stream[stream["scenario"] == scenario].reset_index(drop=True)
start = st.sidebar.number_input("Start at slot", 0, max(len(sub) - 1, 0), 0, 50)
# Navigation helper only: the truth columns are used to find where an example begins; the status that is
# displayed always comes from the gate and the classifier, never from the truth columns.
DISPLAY_ORDER = ["(use the slot above)", "Normal", "Soiling fault", "DC wiring fault", "Inverter fault",
                 "Packet loss", "Communication noise", "Reduced sampling", "Communication outage"]
jump = st.sidebar.selectbox("Jump to the first example of", DISPLAY_ORDER)
if jump != DISPLAY_ORDER[0] and "truth_Label" in sub.columns and "truth_channel" in sub.columns:
    k = DISPLAY_ORDER.index(jump)
    hit = sub.index[sub["truth_Label"] == k - 1] if k <= 4 else sub.index[sub["truth_channel"] == k - 4]
    if len(hit):
        start = int(max(hit[0] - (0 if k <= 4 else 6), 0))      # a few slots of context so the receiver rules can fire
    else:
        st.sidebar.warning("This state does not occur in the selected scenario.")
rows_to_show = st.sidebar.slider("Slots to replay", 50, 1000, 300, 50)
speed = st.sidebar.slider("Replay speed (seconds per slot)", 0.0, 2.0, 0.1, 0.05)
st.sidebar.metric("Classifier accuracy (held-out weeks)", f"{acc * 100:.2f}%")
st.sidebar.metric("With 12-record confirmation", f"{acc_conf * 100:.2f}%")
st.sidebar.caption(f"{n_hold} held-out records; weeks of the replay stream are excluded from training.")
st.sidebar.write("The receiver rules decide the channel state from sequence of arrivals and the CRC flag; "
                 "the truth columns are used only to score the detector.")
run = st.checkbox("Start Replay")

CH_STYLE = {
    1: ("PACKET LOSS FAULT", "blue", "Packets were lost during data transmission. Therefore, readings are not reliable."),
    2: ("COMMUNICATION NOISE FAULT", "purple", "Frame failed the CRC check. Therefore, readings are not reliable."),
    3: ("DOWNSAMPLING FAULT", "teal", "Sampling rate is reduced. Therefore, readings are not reliable."),
    4: ("COMMUNICATION OUTAGE", "blue", "Communication link is interrupted. No data is received from the PV station."),
}
EL_STYLE = {0: ("NORMAL", "green"), 1: ("SOILING FAULT", "orange"), 2: ("DC WIRING FAULT", "red"), 3: ("INVERTER FAULT", "darkred")}

status_box = st.empty()
c1, c2, c3, c4 = st.columns(4)
power_box, ghi_box, temp_box, eff_box = c1.empty(), c2.empty(), c3.empty(), c4.empty()
c5, c6, c7, c8 = st.columns(4)
loss_box, noise_box, down_box, out_box = c5.empty(), c6.empty(), c7.empty(), c8.empty()
chart_cols = st.columns(2)
chart_g_box, chart_p_box = chart_cols[0].empty(), chart_cols[1].empty()
progress_box = st.empty()
score_box = st.empty()


def draw_series(box, series, title):
    """Titled line chart with a zero baseline; both charts share the same x-axis (replay record index). Falls back to the plain chart if Altair is missing."""
    try:
        import altair as alt
        df = pd.DataFrame({"record": range(len(series)), "value": series})
        ch = (alt.Chart(df).mark_line()
              .encode(x=alt.X("record:Q", title="Replay record (30-min slots)"),
                      y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=True)))
              .properties(title=title, height=260))
        box.altair_chart(ch, use_container_width=True)
    except Exception:
        box.line_chart(pd.DataFrame({title: series}))


def num(v):
    return "No Data" if pd.isna(v) else round(float(v), 2)


if run:
    proc = C.StreamProcessor(model)
    hist_p, hist_g = [], []
    recent = deque(maxlen=50)           # (received, crc_ok) of the last 50 slots
    tally = {"n": 0, "ch_ok": 0, "imp": 0, "imp_det": 0, "acc_n": 0, "acc_ok": 0}
    last_arrival = None
    end = min(start + rows_to_show, len(sub))
    total = max(end - start, 1)
    for i in range(start, end):
        r = sub.iloc[i]
        ts = C.parse_ts(pd.Series([r["Time stamp"]])).iloc[0]
        received, crc_ok = int(r["received"]), int(r["crc_ok"])
        vals = [r[C.SRC_COLS[k]] for k in C.RAW]
        out = proc.step(int(r["slot"]), ts, received, crc_ok, vals if received and crc_ok else None)
        recent.append((received, crc_ok))
        if received:
            last_arrival = int(r["slot"])
        ch = out["channel"]

        if ch != 0:
            title, color, msg = CH_STYLE[ch]
            status_box.markdown(
                f"<h2 style='color:{color}; text-align:center;'>CURRENT STATUS: {title}</h2>"
                f"<h4 style='color:{color}; text-align:center;'>{msg}</h4>", unsafe_allow_html=True)
            for box, lab in ((power_box, "Power (kW)"), (ghi_box, "GHI (W/m²)"), (temp_box, "Temperature (°C)"),
                             (eff_box, "Inverter Efficiency (%)")):
                box.metric(lab, "No Data")
            hist_p.append(np.nan); hist_g.append(np.nan)
        else:
            label, color = EL_STYLE.get(out["pred"], ("UNKNOWN", "gray"))
            conf_label = EL_STYLE.get(out["confirmed"], ("UNKNOWN", "gray"))[0]
            status_box.markdown(
                f"<h2 style='color:{color}; text-align:center;'>CURRENT STATUS: {label}</h2>"
                f"<h4 style='text-align:center;'>Confirmed over the last 12 records: {conf_label}</h4>", unsafe_allow_html=True)
            power_box.metric("Power (kW)", num(vals[C.RAW.index("P")]))
            ghi_box.metric("GHI (W/m²)", num(vals[C.RAW.index("GHI")]))
            temp_box.metric("Temperature (°C)", num(vals[C.RAW.index("T")]))
            eff_box.metric("Inverter Efficiency (%)", num(vals[C.RAW.index("eff")]))
            hist_p.append(float(vals[C.RAW.index("P")])); hist_g.append(float(vals[C.RAW.index("GHI")]))

        miss = sum(1 for a, _ in recent if not a)
        loss_box.metric("Missing slots, last 50 (%)", round(100.0 * miss / len(recent), 1))
        noise_box.metric("CRC failures, last 50", sum(1 for a, b in recent if a and not b))
        gap = (int(r["slot"]) - last_arrival) if last_arrival is not None else 0
        down_box.metric("Slots since last frame", gap)
        out_box.metric("Outage", "Yes" if ch == 4 else "No")
        draw_series(chart_g_box, hist_g, "Irradiance GHI (W/m²)")
        draw_series(chart_p_box, hist_p, "System power (kW)")
        progress_box.progress((i - start + 1) / total)

        if "truth_channel" in sub.columns:                         # scoring only; never used by the detector
            tally["n"] += 1
            t_ch = int(r["truth_channel"])
            if t_ch > 0:
                tally["imp"] += 1; tally["imp_det"] += int(ch > 0)
            if ch == 0 and "truth_Label" in sub.columns:
                tally["acc_n"] += 1; tally["acc_ok"] += int(out["pred"] == int(r["truth_Label"]))
            score_box.write(
                f"Scoring against truth columns - slots: {tally['n']}, impaired slots flagged: {tally['imp_det']}/{tally['imp']}, "
                f"electrical accuracy on accepted slots: "
                f"{(100.0 * tally['acc_ok'] / tally['acc_n']) if tally['acc_n'] else float('nan'):.1f}%")
        time.sleep(speed)
else:
    st.info("Press Start Replay to begin monitoring.")

with st.expander("Stream preview"):
    st.dataframe(sub.head(40), use_container_width=True)

with st.expander("How the channel state is decided"):
    st.write("0 = OK (frame arrived and CRC passed); 1 = packet loss (fewer than 5 consecutive missing slots); "
             "2 = noise (frame arrived but the CRC failed); 3 = reduced sampling (regular arrival interval longer than one slot); "
             "4 = outage (5 or more consecutive missing slots). Any state other than 0 shows No Data and the "
             "electrical classifier is not run.")
