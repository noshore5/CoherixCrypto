import datetime
import numpy as np
from scipy.signal import detrend
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import fcwt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['figure.facecolor'] = '#181818'
plt.rcParams['axes.facecolor'] = '#232323'
plt.rcParams['axes.edgecolor'] = '#f0f0f0'
plt.rcParams['axes.labelcolor'] = '#f0f0f0'
plt.rcParams['xtick.color'] = '#f0f0f0'
plt.rcParams['ytick.color'] = '#f0f0f0'
plt.rcParams['text.color'] = '#f0f0f0'
plt.rcParams['savefig.facecolor'] = '#181818'
plt.rcParams['savefig.edgecolor'] = '#181818'


API_KEY = 'jjeryxeZXNkBhTEQF0SDj8uBBI_N1dBM'

def pull_data(ticker, api_key):
    years = 1
    end_datetime = datetime.datetime(2024,8,1)
    start_datetime = end_datetime - datetime.timedelta(days=round(365*years))
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_datetime.date()}/{end_datetime.date()}?adjusted=true&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    if 'results' not in data:
        raise ValueError(f"Error fetching data: {data}")
    times = np.array([entry['t'] for entry in data['results']])
    close = np.array([entry['c'] for entry in data['results']])
    detrended_close = detrend(close)
    return times, close, detrended_close, data

def plot_stocks(sig1, sig2, times1=None, times2=None, ticker1='Stock 1', ticker2='Stock 2'):
    plt.figure(figsize=(10, 3))
    sig1_pct = 100 * (sig1 - np.min(sig1)) / (np.max(sig1) - np.min(sig1))
    sig2_pct = 100 * (sig2 - np.min(sig2)) / (np.max(sig2) - np.min(sig2))
    if times1 is not None and times2 is not None:
        plt.plot(times1, sig1_pct, label=f'{ticker1} (Target)', color='#4e79a7')
        plt.plot(times2, sig2_pct, label=f'{ticker2} (Reference)', color='#f28e2b')
    else:
        plt.plot(sig1_pct, label=f'{ticker1} (Target)', color='#4e79a7')
        plt.plot(sig2_pct, label=f'{ticker2} (Reference)', color='#f28e2b')
    plt.gca().set_facecolor('#232323')
    plt.legend(loc='upper right')
    plt.ylabel('Normalized (%)')
    plt.xticks([])
    plt.title('Input Stock Time Series (Detrended)')
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.close(fig1)
    return fig1

import matplotlib.pyplot as plt
import numpy as np
import datetime

def plot_stocks_close(close1, close2, times1=None, times2=None, ticker1='Stock 1', ticker2='Stock 2'):
    plt.figure(figsize=(14, 8))

    # Normalize both series to show relative change from their first value
    close1_rel = 100 * (close1 - close1[0]) / close1[0]
    close2_rel = 100 * (close2 - close2[0]) / close2[0]

    # Convert Polygon timestamps (ms since epoch) to datetime if needed
    def convert_times(times):
        if times is not None and np.issubdtype(np.array(times).dtype, np.integer):
            return [datetime.datetime.fromtimestamp(t / 1000) for t in times]
        return times

    times1 = convert_times(times1)
    times2 = convert_times(times2)

    # Plot
    if times1 is not None and times2 is not None:
        plt.plot(times1, close1_rel, label=f'{ticker1}', color='#4e79a7')
        plt.plot(times2, close2_rel, label=f'{ticker2}', color='#f28e2b')
        plt.xlabel('Time',fontsize=20)
    else:
        plt.plot(close1_rel, label=f'{ticker1} (Target)', color='#4e79a7')
        plt.plot(close2_rel, label=f'{ticker2} (Reference)', color='#f28e2b')
        plt.xlabel('Index')
    fontsize = 20
    # Style
    plt.gca().set_facecolor('#232323')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.ylabel('Relative Change (%)', fontsize=fontsize+1)
    #plt.title('Input Stock Time Series (Relative Close)', fontsize=16)
    plt.tight_layout()

    fig = plt.gcf()
    plt.close(fig)
    return fig


def workflow(sig1, sig2, bounds, phase=True, density=20, ticker1='Stock 1', ticker2='Stock 2', close1=None, close2=None, times1=None, times2=None):
    sampling_rate = 250
    highest = bounds[0]
    lowest = bounds[1]
    nfreqs = 100
    freqs, coeffs1 = fcwt.cwt(sig1, sampling_rate, lowest, highest, nfreqs, nthreads=4, scaling='log')
    _, coeffs2 = fcwt.cwt(sig2, sampling_rate, lowest, highest, nfreqs, nthreads=4, scaling='log')
    S1 = np.abs(coeffs1) ** 2
    S2 = np.abs(coeffs2) ** 2
    S12 = coeffs1 * np.conj(coeffs2)
    def smooth(data, sigma=(2, 2), mode='nearest'):
        return gaussian_filter(data, sigma=sigma, mode=mode)
    S1_smooth = smooth(S1)
    S2_smooth = smooth(S2)
    S12_smooth = smooth(np.abs(S12) ** 2)
    coh = S12_smooth / (np.sqrt(S1_smooth) * np.sqrt(S2_smooth))
    stock_fig = plot_stocks(sig1, sig2, times1, times2, ticker1, ticker2)
    stock_close_fig = None
    if close1 is not None and close2 is not None:
        stock_close_fig = plot_stocks_close(close1, close2, times1, times2, ticker1, ticker2)
    coherence_fig = coherence_plot_with_arrows(coh, freqs, S12, density, sig1=sig1, sig2=sig2, ticker1=ticker1, ticker2=ticker2)
    return freqs, S12, coh, coherence_fig, stock_fig, stock_close_fig

def coherence_plot_with_arrows(coherence, freqs, S12, density=20, sig1=None, sig2=None, ticker1='Stock 1', ticker2='Stock 2'):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    font_size = 20  # <== Change this to control all font sizes

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Top: time series
    ax0 = fig.add_subplot(gs[0])
    if sig1 is not None and sig2 is not None:
        ax0.plot(sig1, label=f'{ticker1}', color='#4e79a7')
        ax0.plot(sig2, label=f'{ticker2}', color='#f28e2b')
        ax0.set_facecolor('#232323')
        ax0.legend(loc='upper right', fontsize=font_size)
        ax0.set_ylabel('Amplitude', fontsize=font_size)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title('Input Time Series (Detrended)', fontsize=font_size)

    # Bottom: coherence plot
    ax = fig.add_subplot(gs[1])
    extent = [0, coherence.shape[1], len(freqs), 0]

    norm_coh = (coherence - coherence.min()) / (coherence.max() - coherence.min())
    phase = np.angle(S12)
    U = np.cos(phase)
    V = np.sin(phase)
    X, Y = np.meshgrid(np.arange(coherence.shape[1]), np.arange(coherence.shape[0]))
    mask = norm_coh > .4
    indices = np.argwhere(mask)
    subsample_rate = max(1, extent[1] // density)
    subsampled_indices = indices[::subsample_rate]
    x_sub = [X[i, j] for i, j in subsampled_indices]
    y_sub = [Y[i, j] for i, j in subsampled_indices]
    u_sub = [U[i, j] for i, j in subsampled_indices]
    v_sub = [V[i, j] for i, j in subsampled_indices]

    y_values = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), len(freqs))
    subsampled_indices = np.linspace(0, len(y_values) - 1, 20, dtype=int)
    subsampled_y_values = y_values[subsampled_indices]

    im = ax.imshow(norm_coh, aspect='auto', extent=extent, cmap='inferno')
    ax.quiver(x_sub, y_sub, u_sub, v_sub, color='#cccccc', scale=20, headwidth=4, headlength=5, headaxislength=4)

    ax.set_yticks(subsampled_indices)
    ax.set_yticklabels(subsampled_y_values.round(2), fontsize=font_size-4)
    ax.set_xticks(np.linspace(0, coherence.shape[1], 5))
    ax.tick_params(axis='x', labelsize=font_size)
    ax.set_ylabel('Frequency (Per Year)', fontsize=font_size)
    ax.set_xlabel('Time (Days)', fontsize=font_size)

    fig.tight_layout()
    return fig

    ax.set_xlabel('Time (Days)', fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

def coherence_plot(coh, freqs, sampling_rate, sig1=None, sig2=None):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Top: time series
    ax0 = fig.add_subplot(gs[0])
    if sig1 is not None and sig2 is not None:
        ax0.plot(sig1, label='Stock 1', color='#cccccc')
        ax0.plot(sig2, label='Stock 2', color='#f0f0f0')
        ax0.set_facecolor('#232323')
        ax0.legend()
        ax0.set_ylabel('Value')
        ax0.set_xticks([])
        ax0.set_title('Stock Time Series')

    # Bottom: coherence plot
    ax = fig.add_subplot(gs[1])
    y_values = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), len(freqs))
    subsampled_indices = np.linspace(0, len(y_values) - 1, 20, dtype=int)
    subsampled_y_values = y_values[subsampled_indices]
    extent = [0, coh.shape[1]/sampling_rate, len(freqs), 0]
    im = ax.imshow(coh, aspect='auto', extent=extent, cmap='cividis')
    ax.set_yticks(subsampled_indices)
    ax.set_yticklabels([f"{y:.1f}" for y in subsampled_y_values])
    fig.colorbar(im, ax=ax, label='Coherence')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Frequency (per year)')
    ax.set_title('Wavelet Coherence')
    fig.tight_layout()
    return fig
