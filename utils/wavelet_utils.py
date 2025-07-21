# pull data from polygon
import datetime
import numpy as np
from scipy.signal import detrend
import requests
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy.io.wavfile as wav
import fcwt
#import ipywidgets as widget
API_KEY = 'jjeryxeZXNkBhTEQF0SDj8uBBI_N1dBM'

def pull_data(ticker, api_key):
    years = 1
    # Define the date range
    end_datetime = datetime.datetime(2024,8,1)
    start_datetime = end_datetime - datetime.timedelta(days = round(365*years))
    # Polygon API endpoint for aggregate bars
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_datetime.date()}/{end_datetime.date()}?adjusted=true&apiKey={api_key}"
    # Fetch the data
    response = requests.get(url)
    data = response.json()
    if 'results' not in data:
        raise ValueError(f"Error fetching data: {data}")
    # Convert data to arrays
    times = np.array([entry['t'] for entry in data['results']])  # Timestamps
    close = np.array([entry['c'] for entry in data['results']])  # Closing prices
    # Detrend closing prices
    detrended_close = detrend(close)
    return times, close, detrended_close, data
'''
# Example usage:
ticker = 'CL'
times, close, detrended, df = pull_data(ticker, API_KEY)
_, close2, detrended2, df2 = pull_data('ES', API_KEY)
'''


# Wavelet coherence function
def wavelet_coherence(signal1, signal2, highest, lowest, nfreqs, frame_rate):
    freqs, coeffs1 = fcwt.cwt(signal1, frame_rate, lowest, highest, nfreqs, nthreads=4)
    freqs, coeffs2 = fcwt.cwt(signal2, frame_rate, lowest, highest, nfreqs, nthreads=4)
    S1 = np.abs(coeffs1) ** 2
    S2 = np.abs(coeffs2) ** 2
    S12 = coeffs1 * np.conj(coeffs2)
    def smooth(data, sigma=(2,2), mode='nearest'):
        return gaussian_filter(data, sigma=sigma, mode=mode)
    S1_smoothed = smooth(S1)
    S2_smoothed = smooth(S2)
    S12_smoothed = smooth(np.abs(S12) ** 2)
    coherence = S12_smoothed / (np.sqrt(S1_smoothed) * np.sqrt(S2_smoothed))
    return coherence, freqs, [coeffs1, coeffs2], S12
'''
lowest = 2 # days
highest = 100 # days
coherence, freqs, coeffs, S12 = wavelet_coherence(detrended, detrended2, 1/lowest, 1/highest, 100, 1)
'''
def coherence_plot_with_arrows(coherence, freqs, S12, density=20):
    extent = [0, coherence.shape[1], len(freqs), 0]
    norm_coh = (coherence - coherence.min()) / (coherence.max() - coherence.min())
    phase = np.angle(S12)
    U = np.cos(phase)
    V = np.sin(phase)
    X, Y = np.meshgrid(np.arange(coherence.shape[1]), np.arange(coherence.shape[0]))
    mask = norm_coh > .4
    indices = np.argwhere(mask)
    subsample_rate = extent[1] // density
    subsampled_indices = indices[::subsample_rate]
    x_sub = [X[i, j] for i, j in subsampled_indices]
    y_sub = [Y[i, j] for i, j in subsampled_indices]
    u_sub = [U[i, j] for i, j in subsampled_indices]
    v_sub = [V[i, j] for i, j in subsampled_indices]
    y_values = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), len(freqs))
    subsampled_indices = np.linspace(0, len(y_values) - 1, 20, dtype=int)
    subsampled_y_values = y_values[subsampled_indices]
    extent = [0, coherence.shape[1], len(freqs), 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(norm_coh, aspect='auto', extent=extent, cmap='Blues')
    ax.quiver(x_sub, y_sub, u_sub, v_sub, color='black', scale=20, headwidth=4, headlength=5, headaxislength=4)
    dt = 1 / 250
    ax.set_yticks(subsampled_indices)
    ax.set_yticklabels(subsampled_y_values.round(2))
    ax.set_xticks(np.linspace(0, coherence.shape[1], 5))
    ax.set_ylabel('Frequency (Per Year)', fontsize=14)
    ax.set_xlabel('Time (Days)', fontsize=14)
    fig.colorbar(im, ax=ax)
    return fig

# --- WAVELET COHERENCE FUNCTION ---
def wavelet_coherence(signal1, signal2, highest_freq, lowest_freq, nfreqs, sampling_rate):
    freqs, coeffs1 = fcwt.cwt(signal1, sampling_rate, lowest_freq, highest_freq, nfreqs, nthreads=4, scaling='log')
    _, coeffs2 = fcwt.cwt(signal2, sampling_rate, lowest_freq, highest_freq, nfreqs, nthreads=4, scaling='log')
    S1 = np.abs(coeffs1) ** 2
    S2 = np.abs(coeffs2) ** 2
    S12 = coeffs1 * np.conj(coeffs2)
    def smooth(data, sigma=(2, 2), mode='nearest'):
        return gaussian_filter(data, sigma=sigma, mode=mode)
    S1_smooth = smooth(S1)
    S2_smooth = smooth(S2)
    S12_smooth = smooth(np.abs(S12) ** 2)
    coherence = S12_smooth / (np.sqrt(S1_smooth) * np.sqrt(S2_smooth))
    return coherence, freqs

def coherence(coeffs1, coeffs2, freqs):
    S1 = np.abs(coeffs1) ** 2
    S2 = np.abs(coeffs2) ** 2
    S12 = coeffs1 * np.conj(coeffs2)
    def smooth(data, sigma=(2, 2), mode='nearest'):
        return gaussian_filter(data, sigma=sigma, mode=mode)
    S1_smooth = smooth(S1)
    S2_smooth = smooth(S2)
    S12_smooth = smooth(np.abs(S12) ** 2)
    coh = S12_smooth / (np.sqrt(S1_smooth) * np.sqrt(S2_smooth))
    return coh, freqs, S12

def transform(signal1, frame_rate, highest, lowest):
    nfreqs = 100
    freqs, coeffs1 = fcwt.cwt(signal1, frame_rate, lowest, highest, nfreqs, nthreads=4, scaling='log')
    return coeffs1, freqs

def transform_plot(signal, frame_rate, highest, lowest):
    coeffs1, freqs = transform(signal, frame_rate, highest, lowest)
    y_values = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), len(freqs))
    subsampled_indices = np.linspace(0, len(y_values) - 1, 20, dtype=int)
    subsampled_y_values = y_values[subsampled_indices]
    extent = [0, len(signal)/frame_rate, len(freqs), 0]
    plt.figure(figsize=(10, 5))
    plt.imshow(np.abs(coeffs1), aspect='auto', extent=extent, cmap='jet')
    plt.yticks(ticks=subsampled_indices, labels=[f"{y:.1f}" for y in subsampled_y_values])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Magnitude')
    cbar.set_ticks([])
    plt.xlabel('Time (Years)')
    plt.ylabel('Frequency (per year)')
    plt.tight_layout()
    plt.show()
    return coeffs1, freqs

def coherence_plot(coh, freqs, sampling_rate):
    y_values = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), len(freqs))
    subsampled_indices = np.linspace(0, len(y_values) - 1, 20, dtype=int)
    subsampled_y_values = y_values[subsampled_indices]
    extent = [0, coh.shape[1]/sampling_rate, len(freqs), 0]
    plt.figure(figsize=(10, 5))
    plt.imshow(coh, aspect='auto', extent=extent, cmap='Blues')
    plt.yticks(ticks=subsampled_indices, labels=[f"{y:.1f}" for y in subsampled_y_values])
    plt.colorbar(label='Coherence')
    plt.xlabel('Time (years)')
    plt.ylabel('Frequency (per year)')
    plt.title('Wavelet Coherence')
    plt.tight_layout()
    plt.show()

def workflow(sig1, sig2, bounds, phase=True, density=20):
    sampling_rate = 250 # Assuming days per year
    highest = bounds[0]
    lowest = bounds[1]
    t1, _ = transform(sig1, 250, highest, lowest)
    t2, freqs = transform(sig2, 250, highest, lowest)
    coh, _, S12 = coherence(t1, t2, freqs)
    if phase:
        fig = coherence_plot_with_arrows(coh, freqs, S12)
    else:
        coherence_plot(coh, freqs, 250)
    return freqs, S12, coh,fig

'''
# --- Widgets ---
ticker1 = widgets.Text(description='Stock 1:')
ticker2 = widgets.Text(description='Stock 2:')
upper_bound = widgets.IntSlider(value=125, min=20, max=250, step=5, description='Max Period:')
lower_bound = widgets.IntSlider(value=8, min=2, max=50, step=1, description='Min Period:')
phase_toggle = widgets.Checkbox(value=True, description='Show Phase Arrows')
run_button = widgets.Button(description='Run Analysis')
output = widgets.Output()

def on_run_clicked(b):
    with output:
        clear_output(wait=True)
        sig1 = pull_data(ticker1.value, key)[2]
        sig2 = pull_data(ticker2.value, key)[2]
        bounds = (upper_bound.value, lower_bound.value)
        workflow(sig1, sig2, bounds, phase=phase_toggle.value, density=20)

run_button.on_click(on_run_clicked)

ui = widgets.VBox([
    widgets.HBox([ticker1, ticker2]),
    upper_bound,
    lower_bound,
    phase_toggle,
    run_button,
    output
])

display(ui)
'''