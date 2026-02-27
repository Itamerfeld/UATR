
# Harmonics Detector (STFT + Comb + Viterbi) with DEMON Cross-check

## What this is
A compact, production-friendly prototype to detect and track harmonic line sets in underwater acoustic data.
It uses:
- STFT + median whitening to emphasize lines
- Comb-scoring of candidate f0 bins per frame
- Viterbi track-before-detect to obtain a smooth f0 trajectory
- Optional DEMON (envelope) spectrum cross-check for propeller/shaft tones

No exotic dependencies; easy to slot into real-time pipelines.

## Files
- `harmonics_detector.py` — the detector module
- `demo_harmonics_detector.py` — generates a synthetic example and runs the detector
- Outputs: `spectrogram_f0.png`, `comb_map.png`, `f0_track_vs_truth.png`, `harm_mask.png`, `demon_peaks.png` (optional), `f0_track.json`

## How to run
```bash
python demo_harmonics_detector.py
```

## Where to customize
- `DetectorParams`: f0 range, comb bandwidth, number of harmonics, Viterbi costs
- `STFTParams`: n_fft/hop (e.g., 4096/1024 for 96–192 kHz data is a good start)
- `DEMONParams`: if you expect propeller modulation, tune the bandpass and decimation

## Notes
- If you want an even sharper TFR, you can replace the STFT with a synchrosqueezed transform; the rest of the pipeline (comb + Viterbi + mask) stays the same.
- For very weak harmonics, increase `band_hz` a bit and `smoothing_alpha` to stabilize the comb map.
