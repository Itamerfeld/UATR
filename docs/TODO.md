TODO (Research):

- ~~Add sea noise (recorded) to signale, show effect on F0 plot.
        - take first 2000s of 1000m-experiment-recording as noise
        - take fist 2000s of 600m-experiment-recording as signal~~

- ~~Add sea noise to AUV recording~~

- ~~Add plot for AUV / DPV, with noise only~~

- ~~DONE: redo all plots as .pdf~~

- Phase estimation instead of spectrogram bins
        - review and test nb90 with gemini suggestions.
        - implement the preffered method in s2g detector utils
        - recreate plots for nb85, nb86, nb88, nb89 with new estimation method.

- ~~Reference method
        - review and test 'dev/detectors/detectors_gpt' with gpt suggestions
        - implement resulting detectors in utils
        - recreate plots for nb85, nb86, nb88, nb89 with new reference algorithm.~~

- refactor detectors into 'feature extraction' static functions

TODO (Simulation):

- 
---
# Plots for thesis:

### Simulation:
1. set SNR choose Q
2. set Q compare SNR
3. set SNR + Q choose K metric
4. set SNR + Q + K compare to reference methods

### Experiment:
1. DPV (f0 comparisone scores)
2. DPV w. different fft window (only s2g score)
3. AUV (f0 comparisone scores)
4. only noise (f0 comparisone scores)

---
# Writing:

### Structure:
- Introduction
        - motivation
        - challenges
        - current approaches
        - my method + contributions + main results
        - paper structure
- Literature Review
        - marine vessel detection
- System Model
        - setup and assumptions
        - Preliminary
                - graph evaluation methods (metrics)
                - phase of shipping noise
- Method
        - key idea + diagram
        - Detection metric (why phase transitions)
        - phase transition graph creation
        - graph evaluation
        - phase estimation
        - ...
- Results
        - Simulation
                - setup
                - results
        - Experiment
                - DPV
                - AUV
- Discussion