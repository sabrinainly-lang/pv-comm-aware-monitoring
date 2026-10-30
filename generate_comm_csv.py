"""Generate the communication-fault stream CSV that the dashboard replays.
Usage: python generate_comm_csv.py <training_dataset.csv> <out_stream.csv> [seed]
Receiver-visible columns: scenario, slot, Time stamp, received, crc_ok and the seven measurements
(blank when the frame never arrived; corrupted values when the CRC failed).
Ground-truth columns (truth_Label, truth_channel) are used only to score the detector, never by it."""
import sys
import pandas as pd
import pvmon_core as C

src, out = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
d = C.load_clean(src); H = C.holdout_stream_records(d)
frames = [C.make_stream_frame(H, sc, seed) for sc in ['clean'] + C.SCENARIOS + ['mixed']]
S = pd.concat(frames, ignore_index=True)
S.to_csv(out, index=False)
print(out, S.shape, S.groupby('scenario').truth_channel.apply(lambda s: round((s > 0).mean(), 3)).to_dict())
