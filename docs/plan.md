# Development Plan: Cyclostationary Harmonic Structure Extraction

## Overview
This plan outlines the development of a system to extract harmonic structure from cyclostationary signals in underwater acoustic target recognition, specifically addressing three core requirements:
1. Finding the base frequency of the signal
2. Identifying which harmonic series are dominant 
3. Computing energy ratios between harmonics

## 1. Base Frequency Detection

### Approach: Multi-Method Fundamental Frequency (F0) Estimation

**Primary Methods:**
- **Cyclic Autocorrelation Peak Detection**: Find dominant cyclic frequency α₀
- **Spectral Correlation Density Analysis**: Identify F0 from S_x^α(f) peaks  
- **Enhanced Autocorrelation Function (EACF)**: Robust to noise and harmonics

### Implementation Strategy:
```python
class BaseFrequencyDetector:
    def __init__(self, fs=192000, freq_range=(1, 1000)):
        self.fs = fs
        self.freq_range = freq_range
        
    def cyclic_autocorr_f0(self, signal, alpha_max=100):
        """Estimate F0 from cyclic autocorrelation peaks"""
        # Compute R_x^α(0) for α ∈ [0, α_max]
        # Peak at α = F0 indicates fundamental cyclostationarity
        
    def spectral_correlation_f0(self, signal):
        """Find F0 from spectral correlation density maxima"""
        # S_x^α(f) analysis - peak at (α=F0, f=F0) 
        
    def enhanced_autocorr_f0(self, signal):
        """Robust F0 via enhanced autocorrelation"""
        # Remove harmonics influence using cepstral analysis
        # Peak picking with parabolic interpolation
        
    def multi_method_consensus(self, signal):
        """Combine all methods with confidence weighting"""
        # Weighted voting across estimation methods
        # Return F0 + confidence interval
```

### Underwater Acoustics Adaptations:
- **Doppler Tracking**: Compensate for F0 shifts due to relative motion
- **Noise Robustness**: Use median filtering and outlier rejection
- **Temporal Continuity**: Kalman filtering for smooth F0 tracking

## 2. Dominant Harmonic Series Identification

### Approach: Spectral Peak Analysis with Harmonic Matching

**Core Algorithm:**
1. **Spectral Peak Detection**: Find all significant spectral peaks
2. **Harmonic Template Matching**: Test integer multiples of candidate F0s
3. **Series Strength Scoring**: Quantify harmonic series dominance
4. **Multiple Series Detection**: Identify overlapping harmonic structures

```python
class HarmonicSeriesAnalyzer:
    def __init__(self, max_harmonics=20, peak_threshold=-20):
        self.max_harmonics = max_harmonics
        self.peak_threshold = peak_threshold  # dB below max
        
    def detect_spectral_peaks(self, spectrum):
        """Find all significant peaks in power spectrum"""
        # Peak detection with minimum separation
        # Amplitude thresholding relative to noise floor
        
    def generate_harmonic_templates(self, f0_candidates):
        """Create harmonic series templates for each F0"""
        # Templates: [F0, 2*F0, 3*F0, ..., N*F0]
        # Include frequency tolerance bands
        
    def match_peaks_to_harmonics(self, peaks, templates):
        """Match detected peaks to harmonic templates"""
        # For each template, count matching peaks
        # Score = (matched_peaks / total_harmonics) * energy_weight
        
    def rank_harmonic_series(self, matching_scores):
        """Rank series by dominance strength"""
        # Dominant series = highest matching score
        # Secondary series = significant but lower scores
        
    def identify_dominant_series(self, signal, f0_estimate):
        """Complete pipeline for harmonic series identification"""
        spectrum = self.compute_power_spectrum(signal)
        peaks = self.detect_spectral_peaks(spectrum)
        templates = self.generate_harmonic_templates([f0_estimate])
        scores = self.match_peaks_to_harmonics(peaks, templates)
        return self.rank_harmonic_series(scores)
```

### Advanced Features:
- **Multiple F0 Detection**: Handle signals with multiple periodic sources
- **Inharmonic Series**: Account for slight deviations from perfect integer ratios
- **Time-Varying Analysis**: Track harmonic series evolution over time

## 3. Energy Ratio Computation Between Harmonics

### Approach: Precise Energy Measurement and Ratio Analysis

**Energy Computation Methods:**
- **Narrowband Integration**: Precise energy in frequency bins around each harmonic
- **Spectral Line Tracking**: Follow harmonic evolution over time
- **Power Spectral Density Integration**: Account for spectral spreading

```python
class HarmonicEnergyAnalyzer:
    def __init__(self, integration_bandwidth=2.0):
        self.bandwidth = integration_bandwidth  # Hz around each harmonic
        
    def compute_harmonic_energies(self, spectrum, frequencies, harmonics):
        """Compute energy for each harmonic in the series"""
        energies = {}
        for n, freq in enumerate(harmonics):
            # Define integration band around harmonic frequency
            f_low = freq - self.bandwidth/2
            f_high = freq + self.bandwidth/2
            
            # Integrate power spectral density
            energy = self.integrate_psd(spectrum, frequencies, f_low, f_high)
            energies[f'H{n+1}'] = energy
            
        return energies
    
    def compute_energy_ratios(self, energies):
        """Calculate energy ratios between harmonics"""
        ratios = {}
        fundamental_energy = energies['H1']
        
        # Harmonic-to-fundamental ratios
        for harmonic, energy in energies.items():
            ratios[f'{harmonic}/H1'] = energy / fundamental_energy
            
        # Adjacent harmonic ratios  
        for i in range(2, len(energies)+1):
            if f'H{i}' in energies and f'H{i-1}' in energies:
                ratios[f'H{i}/H{i-1}'] = energies[f'H{i}'] / energies[f'H{i-1}']
                
        # Total harmonic energy vs fundamental
        total_harmonic = sum(energies[k] for k in energies if k != 'H1')
        ratios['THD'] = total_harmonic / fundamental_energy
        
        return ratios
    
    def temporal_energy_evolution(self, signal, hop_length=1024):
        """Track energy ratios over time"""
        # Sliding window analysis
        # Return time series of energy ratio evolution
```

## 4. Integrated Solution Architecture

```python
class CyclostationaryHarmonicExtractor:
    def __init__(self, fs=192000):
        self.fs = fs
        self.f0_detector = BaseFrequencyDetector(fs)
        self.series_analyzer = HarmonicSeriesAnalyzer()
        self.energy_analyzer = HarmonicEnergyAnalyzer()
        
    def extract_harmonic_structure(self, signal):
        """Complete harmonic analysis pipeline"""
        
        # 1. Find base frequency
        f0, f0_confidence = self.f0_detector.multi_method_consensus(signal)
        
        # 2. Identify dominant harmonic series
        dominant_series = self.series_analyzer.identify_dominant_series(signal, f0)
        
        # 3. Compute energy ratios
        spectrum = np.abs(np.fft.fft(signal))**2
        frequencies = np.fft.fftfreq(len(signal), 1/self.fs)
        
        results = {}
        for series_name, harmonics in dominant_series.items():
            energies = self.energy_analyzer.compute_harmonic_energies(
                spectrum, frequencies, harmonics)
            ratios = self.energy_analyzer.compute_energy_ratios(energies)
            
            results[series_name] = {
                'base_frequency': f0,
                'confidence': f0_confidence,
                'harmonic_frequencies': harmonics,
                'energies': energies,
                'energy_ratios': ratios
            }
            
        return results
```

## 5. Validation & Performance Metrics

### Test Cases:
- **Synthetic Signals**: Known F0 and harmonic structure
- **Scooter Data**: Motor harmonics from Eilat recordings
- **Vessel Data**: Propeller blade rate frequencies from DeepShip

### Performance Metrics:
- **F0 Accuracy**: RMSE between estimated and true fundamental
- **Harmonic Detection Rate**: Precision/recall for harmonic identification
- **Energy Ratio Error**: Accuracy of relative energy measurements

### Expected Output Structure:
```python
# Example result structure
{
    'primary_series': {
        'base_frequency': 23.4,  # Hz
        'confidence': 0.87,
        'harmonic_frequencies': [23.4, 46.8, 70.2, 93.6, 117.0],
        'energies': {'H1': 0.45, 'H2': 0.23, 'H3': 0.12, 'H4': 0.08, 'H5': 0.04},
        'energy_ratios': {
            'H2/H1': 0.51, 'H3/H1': 0.27, 'H4/H1': 0.18,
            'THD': 1.04  # Total Harmonic Distortion
        }
    },
    'secondary_series': {
        # Additional harmonic series if detected
    }
}
```

## 6. Mathematical Foundation

### Cyclostationary Signal Theory:
```
# Cyclic Autocorrelation Function
R_x^α(τ) = lim_{T→∞} (1/T) ∫ x(t+τ/2) x*(t-τ/2) e^{-j2παt} dt

# Spectral Correlation Density  
S_x^α(f) = ∫ R_x^α(τ) e^{-j2πfτ} dτ

# Cyclic Frequency α represents the cyclostationarity rate
```

### Signal Characteristics in Underwater Acoustics:
- **Propeller Signatures**: Blade rate frequency (BRF) and shaft rate frequency (SRF) harmonics
- **Engine Harmonics**: Combustion cycles, mechanical vibrations
- **Scooter Motors**: Brushless motor switching frequencies, rotor harmonics
- **Temporal Cyclicity**: Periodic statistical properties varying with vessel motion patterns

## 7. Implementation Priorities

1. **Phase 1**: Base frequency detection implementation
2. **Phase 2**: Harmonic series identification
3. **Phase 3**: Energy ratio computation
4. **Phase 4**: Integration and validation framework
5. **Phase 5**: Real-time optimization and deployment

## Dependencies

```python
numpy, scipy          # Core numerical computing
matplotlib, seaborn   # Visualization  
librosa              # Audio signal processing
scikit-learn         # ML utilities
numba               # JIT compilation for speed
```

## Performance Targets

- Process 192kHz audio in real-time (< 5ms latency)
- Detect harmonics with <0.1Hz frequency accuracy
- Operate in SNR conditions down to -10dB