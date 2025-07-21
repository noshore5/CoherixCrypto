import pandas as pd
import numpy as np
import fcwt
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend
import matplotlib.pyplot as plt

def load_sp500_djia(filepath):
    df = pd.read_csv(filepath, usecols=['sp500', 'djia'])
    detrended_sp500 = detrend(df['sp500'])
    detrended_djia = detrend(df['djia'])
    df['detrended_sp500'] = detrended_sp500
    df['detrended_djia'] = detrended_djia
    return df

# Example usage:
df = load_sp500_djia('stock_data.csv')


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

def workflow(sig1, sig2, bounds, phase=True, density=20):
    sampling_rate = 250 # Assuming days per year
    highest = bounds[0]
    lowest = bounds[1]
    t1, _ = transform(sig1, 250, highest, lowest)
    t2, freqs = transform(sig2, 250, highest, lowest)
    coh, _, S12 = coherence(t1, t2, freqs)
    if phase:
        fig = coherence_plot_with_arrows(coh, freqs, S12)

    return freqs, S12, coh,fig