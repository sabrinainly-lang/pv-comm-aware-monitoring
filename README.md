# Communication-aware PV fault monitoring - code package

Final model: multilayer perceptron (StandardScaler + MLP 64-32, random_state 42). results_rf.json / replay_results_rf.json keep the earlier Random Forest run for comparison.

Files
- pvmon_core.py   : features, receiver-side channel gate (loss/outage/reduced sampling/CRC noise), alarm confirmation, stream replay
- generate_comm_csv.py : makes the communication-fault stream CSV
- evaluate.py     : week-blocked 5-fold CV, ablation, channel experiments (writes results.json)
- test_replay.py  : replays the stream CSV through the same code as the dashboard
- app.py          : Streamlit dashboard
- Solar_Fault_Dataset_With_Comm_Loss_No_New_Label.csv : training data (SAM)
- Solar_PV_Comm_Stream_Test.csv : generated stream (8 scenarios, seed 42)

Run
    pip install streamlit scikit-learn pandas numpy
    python generate_comm_csv.py Solar_Fault_Dataset_With_Comm_Loss_No_New_Label.csv Solar_PV_Comm_Stream_Test.csv 42
    python test_replay.py Solar_Fault_Dataset_With_Comm_Loss_No_New_Label.csv Solar_PV_Comm_Stream_Test.csv replay_results.json
    python evaluate.py        # see script header for arguments
    streamlit run app.py

Notes: the channel is simulated; the Streamlit UI was not rendered in the authoring environment (logic stub-tested only).

Algorithm comparison (same week-blocked 5-fold protocol):
    python compare_models.py Solar_Fault_Dataset_With_Comm_Loss_No_New_Label.csv compare_results.json
    python mlp_check.py Solar_Fault_Dataset_With_Comm_Loss_No_New_Label.csv mlp_check.json


---- Added for the revised paper (Sections V-C, VI-A, VI-E, VI-F, VI-G) ----
Files
- extra_eval.py        : bootstrap intervals, ablations, onset / recovery / outage tests, false alarms  -> extra_results.json
- sensor_fault_eval.py : invalid sensor values inside a valid frame (gate passes them)                  -> sensor_results.json
- severity_eval.py     : severity sweep + second site; reads sam_scenarios_manifest.csv and sam_runs/    -> severity_results.json
- sam_runs/            : the 21 SAM exports of the severity runs (w1 = Tripoli weather file, w2 = second site)
- sam_config.md, requirements.txt : SAM settings and software versions

Reproduce (seeds: network random_state 42; channel seeds 0-19; bootstrap seed 0, B = 2000)
    pip install -r requirements.txt
    python evaluate.py Solar_Fault_Dataset_With_Comm_Loss_No_New_Label.csv results.json
    python extra_eval.py Solar_Fault_Dataset_With_Comm_Loss_No_New_Label.csv extra_results.json
    python sensor_fault_eval.py Solar_Fault_Dataset_With_Comm_Loss_No_New_Label.csv sensor_results.json
    python severity_eval.py sam_scenarios_manifest.csv severity_results.json
    streamlit run app.py
(see each script header for arguments; the Streamlit UI itself is not unit-tested)
