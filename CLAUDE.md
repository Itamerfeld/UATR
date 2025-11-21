# UATR - Underwater Acoustic Target Recognition

## Project Overview
Marine vesseles, diver propulsion vehicle, and autonomous underwater vehicles classification from underwater acoustic recordings. Analyzing real-world hydrophone data to identify vessel types.

## Current Stack
- Python: numpy, pandas, matplotlib, seaborn, scipy
- Data: Audio files (.wav, .pkf) at 192kHz sample rate
- Analysis: Signal processing, pattern recognition, spectrograms

## Key Directories
- `/docs/` - Documentation, analysis, reviews and papers
  - Analysis.pdf - Project analysis documentation  
  - Survey paper on AI-Based Underwater Acoustic Target Recognition
- `/data/` - Audio recordings and datasets
  - `2024_12_25_scooter_eilat/` - Field recordings of scooter/DPV from Eilat
  - `DeepShip/` - DeepShip dataset for vessel classification
  - `croatia/` - Additional field recordings
  - Individual .wav/.pkf files at 192kHz sample rate
- `/notebooks/` - Jupyter analysis notebooks
  - `nb1.ipynb` - Pattern analysis, windowing, and feature extraction
- `/dev/` - Development utilities and tools

## Domain Context
- **Goal**: Classify ships from acoustic signatures
- **Data**: DeepShip dataset + field recordings
- **Approach**: Window-based pattern analysis and feature extraction

## Data Details
- **Recording Equipment**: icListen hydrophones (RBW6695 series)
- **File Formats**: .wav (audio) and .pkf (proprietary format) 
- **Temporal Resolution**: Minute-by-minute recordings from field sessions
- **Target Classes**: Marine vessels, diver propulsion vehicles (scooters), AUVs

## Analysis Approach
- **Window-based Processing**: Extract patterns from fixed-size windows (6-12 samples)
- **Pattern Averaging**: Compute average patterns across multiple windows
- **Feature Extraction**: Harmonics, spectrograms, temporal patterns
- **Current Focus**: Pattern recognition from scooter recordings in Eilat

## Notes
- Large audio files - process in chunks
- Real-world data includes marine background noise
- Focus on practical signal processing solutions
- Field recordings from real deployments (Eilat, Croatia)