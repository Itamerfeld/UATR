import numpy as np
import plotly.graph_objects as go
from scipy import signal
from scipy.io import wavfile

def create_spectrogram_plotly(audio_data, 
                              sample_rate, 
                              nperseg=8192, 
                              noverlap=None, 
                              window='hann', 
                              title='Spectrogram', 
                              show=True, 
                              colorscale='Viridis', 
                              crop_freq=None, 
                              phase=False):
    """
    Create and display a spectrogram using Plotly.
    
    Parameters:
    -----------
    audio_data : array-like
        Audio signal data
    sample_rate : int
        Sample rate of the audio signal
    nperseg : int, optional
        Length of each segment for STFT (default: 1024)
    noverlap : int, optional
        Number of points to overlap between segments (default: nperseg//8)
    window : str, optional
        Window function to use (default: 'hann')
    title : str, optional
        Title for the plot (default: 'Spectrogram')
    show : bool, optional
        Whether to display the plot (default: True)
    colorscale : str, optional
        Plotly colorscale (default: 'Viridis')
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    
    if noverlap is None:
        noverlap = nperseg // 8
    
    # Compute spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        audio_data, 
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        mode='phase' if phase else 'magnitude'
    )

    # Crop frequencies if specified
    if crop_freq is not None:
        freq_mask = frequencies <= crop_freq
        frequencies = frequencies[freq_mask]
        Sxx = Sxx[freq_mask, :]
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=Sxx_db,
        x=times,
        y=frequencies,
        colorscale=colorscale,
        colorbar=dict(title='Power (dB)')
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        width=1600,
        height=1200
    )
    
    if show:
        fig.show()
    
    return fig

def create_spectrogram_from_file(filepath, title=None, **kwargs):
    """
    Create and display a spectrogram directly from an audio file.
    
    Parameters:
    -----------
    filepath : str
        Path to the audio file (.wav)
    title : str, optional
        Title for the plot (default: filename)
    **kwargs
        Additional arguments passed to create_spectrogram_plotly
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    
    # Read the audio file
    sample_rate, audio_data = wavfile.read(filepath)
    
    # Use filename as title if not provided
    if title is None:
        title = f'Spectrogram - {filepath.split("/")[-1]}'
    
    return create_spectrogram_plotly(audio_data, sample_rate, title=title, **kwargs)


if __name__ == "__main__":
    audio_data = 'data/scooter_example_1_downsample32k.wav'
    create_spectrogram_from_file(audio_data, title='Spectrogram', crop_freq=2000, mode='phase')