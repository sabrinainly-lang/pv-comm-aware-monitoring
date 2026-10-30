# SAM configuration of the simulated plant

- Tool: NREL System Advisor Model (SAM). Main four scenario files: earlier SAM library version; severity runs and the second-site runs: SAM 2026.7.3.
- Model: detailed PV model (PVWatts not used), grid-connected, 30-minute time step.
- Weather file (Tripoli): NSRDB lat_32.89_lon_13.19_msg-iodc_30_2019 (calendar year 2019, 30 min, native resolution; no TMY, no resampling).
- Array: 30 modules, 10 per string, 3 strings; module SunSpark Technology SG7G54M(H)-400 (400 W), bifacial modeling disabled; fixed, tilt 20 deg, south-facing; 12 kW DC.
- Inverter: Fronius International GmbH: Primo GEN24 10.0 208-240 {240V} (CEC database), 10 kW class.
- Losses (normal operation): soiling 5 %, DC wiring 2 %, module mismatch 2 %, diode and connections 0.5 % (total DC loss 4.44 %), AC wiring 1 %.
- Scenarios: soiling loss 25 %; DC wiring loss 10 %; inverter replaced by Fronius Primo GEN24 6.0 Plus (6 kW class) - a capacity-limitation scenario, not a degradation of the same unit.
- Severity runs: soiling 10/15/20/30 %; DC wiring 4/6/8/12 %; maximum AC power of the same inverter at 90/80/70/60 % (files sam_runs/w1_*).
- Second site (Sebha): files sam_runs/w2_*; same plant settings; NSRDB lat_27.04_lon_14.43_msg-iodc_30_2017 (station 805000, elevation 428 m, calendar year 2017, 30 min; GHI 2,221 kWh/m2/yr).
- Time stamps of the original dataset are 2 hours earlier than the stamps of a SAM export of the same weather file (constant offset).
