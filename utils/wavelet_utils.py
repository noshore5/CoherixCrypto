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

def pull_data(ticker, api_key=None, start_ms=None, end_ms=None, num_samples=1000):
    """
    Fetch historical close prices from Binance API.
    :param ticker: Binance symbol, e.g., 'BTCUSDT'
    :param api_key: (Unused, kept for compatibility)
    :param start_ms: Start time in ms since epoch
    :param end_ms: End time in ms since epoch
    :param num_samples: Number of samples to fetch (max 1000 per Binance API)
    :return: times (np.array), close (np.array), detrended_close (np.array), raw_data (list)
    """
    import time

    # Always use the smallest interval possible to maximize resolution
    if start_ms is not None and end_ms is not None:
        total_minutes = (end_ms - start_ms) / 1000 / 60
        # Try to get as close to num_samples as possible
        if num_samples > 500 and total_minutes <= 7 * 24 * 60:
            interval = "5m"
        elif num_samples > 200 and total_minutes <= 30 * 24 * 60:
            interval = "15m"
        elif total_minutes <= 90 * 24 * 60:
            interval = "1h"
        else:
            interval = "1d"
    else:
        interval = "1d"

    url = "https://api.binance.com/api/v3/klines"

    # If not provided, use default 1 year ending at 2024-08-01
    if end_ms is None:
        end_datetime = datetime.datetime(2024, 8, 1)
        end_ms = int(end_datetime.timestamp() * 1000)
    if start_ms is None:
        years = 1
        start_datetime = datetime.datetime.fromtimestamp(end_ms / 1000) - datetime.timedelta(days=round(365 * years))
        start_ms = int(start_datetime.timestamp() * 1000)

    params = {
        "symbol": ticker,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": min(num_samples, 1000)  # Binance max per request
    }
    response = requests.get(url, params=params)
    data = response.json()
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Error fetching data: {data}")

    # Each kline: [open_time, open, high, low, close, volume, close_time, ...]
    times = np.array([entry[0] for entry in data])
    close = np.array([float(entry[4]) for entry in data])
    detrended_close = detrend(close)
    return times, close, detrended_close, data

def plot_stocks(sig1, sig2, times1=None, times2=None, ticker1='Stock 1', ticker2='Stock 2'):
    plt.figure(figsize=(10, 3))
    sig1_pct = 100 * (sig1 - np.min(sig1)) / (np.max(sig1) - np.min(sig1))
    sig2_pct = 100 * (sig2 - np.min(sig2)) / (np.max(sig2) - np.min(sig2))
    ax = plt.gca()
    if times1 is not None and times2 is not None:
        # Convert ms timestamps to datetime for better x-axis
        if np.issubdtype(np.array(times1).dtype, np.integer):
            times1_dt = [datetime.datetime.fromtimestamp(t / 1000) for t in times1]
        else:
            times1_dt = times1
        if np.issubdtype(np.array(times2).dtype, np.integer):
            times2_dt = [datetime.datetime.fromtimestamp(t / 1000) for t in times2]
        else:
            times2_dt = times2
        ax.plot(times1_dt, sig1_pct, label=f'{ticker1} (Target)', color='#4e79a7')
        ax.plot(times2_dt, sig2_pct, label=f'{ticker2} (Reference)', color='#f28e2b')
        ax.set_xlabel('Time')
        plt.xticks(rotation=45)
    else:
        ax.plot(sig1_pct, label=f'{ticker1} (Target)', color='#4e79a7')
        ax.plot(sig2_pct, label=f'{ticker2} (Reference)', color='#f28e2b')
        ax.set_xlabel('Index')
    ax.set_facecolor('#232323')
    ax.legend(loc='upper right')
    ax.set_ylabel('Normalized (%)')
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
    close1_rel = 100 * (close1 - close1[0]) / close1[0]
    close2_rel = 100 * (close2 - close2[0]) / close2[0]
    def convert_times(times):
        if times is not None and np.issubdtype(np.array(times).dtype, np.integer):
            return [datetime.datetime.fromtimestamp(t / 1000) for t in times]
        return times
    times1 = convert_times(times1)
    times2 = convert_times(times2)
    ax = plt.gca()
    if times1 is not None and times2 is not None:
        ax.plot(times1, close1_rel, label=f'{ticker1}', color='#4e79a7')
        ax.plot(times2, close2_rel, label=f'{ticker2}', color='#f28e2b')
        ax.set_xlabel('Time', fontsize=20)
        plt.xticks(rotation=45)
    else:
        ax.plot(close1_rel, label=f'{ticker1} (Target)', color='#4e79a7')
        ax.plot(close2_rel, label=f'{ticker2} (Reference)', color='#f28e2b')
        ax.set_xlabel('Index')
    fontsize = 20
    ax.set_facecolor('#232323')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.legend(loc='upper right', fontsize=fontsize)
    ax.set_ylabel('Relative Change (%)', fontsize=fontsize+1)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close(fig)
    return fig


def workflow(
    sig1, sig2, bounds, phase=True, density=20, ticker1='Stock 1', ticker2='Stock 2',
    close1=None, close2=None, times1=None, times2=None,
    begin=None, end=None, sampling_rate=250
):
    # Remove begin/end slicing, always use full arrays (handled by interpolation in app.py)
    # Use relative price changes for detrended inputs
    sig1_rel = 100 * (sig1 - sig1[0]) / sig1[0]
    sig2_rel = 100 * (sig2 - sig2[0]) / sig2[0]
    # Calculate dt in days
    if times1 is not None and len(times1) > 1:
        dt_seconds = (times1[1] - times1[0]) / 1000
        dt_days = dt_seconds / (60 * 60 * 24)
    else:
        dt_days = 1
    # Nyquist frequency (cycles per day or per year)
    nyquist = 0.5 / dt_days
    # Set frequency axis dynamically
    if dt_days < 1:
        freq_unit = 'cycles/day'
        highest = nyquist
        lowest = max(bounds[1], 1 / (len(sig1) * dt_days))
    else:
        freq_unit = 'cycles/year'
        highest = nyquist * 365.25
        lowest = max(bounds[1], 1 / (len(sig1) * dt_days / 365.25))
    nfreqs = 100
    freqs, coeffs1 = fcwt.cwt(sig1_rel, sampling_rate, lowest, highest, nfreqs, nthreads=4, scaling='log')
    _, coeffs2 = fcwt.cwt(sig2_rel, sampling_rate, lowest, highest, nfreqs, nthreads=4, scaling='log')
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
    coherence_fig = coherence_plot_with_arrows(
        coh, freqs, S12, density,
        sig1=sig1_rel, sig2=sig2_rel, ticker1=ticker1, ticker2=ticker2,
        sampling_rate=sampling_rate, times=times1, freq_unit=freq_unit
    )
    return freqs, S12, coh, coherence_fig, stock_fig, stock_close_fig

def coherence_plot_with_arrows(
    coherence, freqs, S12, density=20, sig1=None, sig2=None, ticker1='Stock 1', ticker2='Stock 2',
    sampling_rate=250, times=None, freq_unit='cycles/day'
):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    font_size = 20

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

    n_times = coherence.shape[1]
    n_freqs = coherence.shape[0]

    # Use index for x, evenly spaced y for frequency axis (not log)
    extent = [0, n_times, n_freqs, 0]
    im = ax.imshow(
        (coherence - coherence.min()) / (coherence.max() - coherence.min()),
        aspect='auto',
        extent=extent,
        cmap='inferno',
        origin='upper'
    )

    # Quiver arrows: sync to the same scale as imshow (index, evenly spaced frequency bins)
    phase = np.angle(S12)
    U = np.cos(phase)
    V = np.sin(phase)
    X, Y = np.meshgrid(np.arange(n_times), np.arange(n_freqs))
    mask = ((coherence - coherence.min()) / (coherence.max() - coherence.min())) > .4
    indices = np.argwhere(mask)
    if len(indices) > 0:
        n_arrows = min(density * density, len(indices))
        step = max(1, len(indices) // n_arrows)
        subsampled_indices_arrows = indices[::step]
        x_sub = [j for i, j in subsampled_indices_arrows]
        y_sub = [i for i, j in subsampled_indices_arrows]
        u_sub = [U[i, j] for i, j in subsampled_indices_arrows]
        v_sub = [V[i, j] for i, j in subsampled_indices_arrows]
        ax.quiver(x_sub, y_sub, u_sub, v_sub, color='#cccccc', scale=30, headwidth=4, headlength=5, headaxislength=4, width=0.003)
    # else: no arrows

    # Y ticks: evenly spaced, label with frequency values (no scientific notation)
    subsampled_indices = np.linspace(0, n_freqs - 1, 20, dtype=int)
    subsampled_y_values = [freqs[i] for i in subsampled_indices]
    ax.set_yticks(subsampled_indices)
    ax.set_yticklabels([f"{y:.3f}" for y in subsampled_y_values], fontsize=font_size)
    freq_label = f'Frequency ({freq_unit})'
    ax.set_ylabel(freq_label, fontsize=font_size)
    ax.set_xlabel('Index', fontsize=font_size)
    plt.tight_layout()
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
    return fig
    return fig
    ax.set_yticks(subsampled_indices)
    ax.set_yticklabels([f"{y:.1f}" for y in subsampled_y_values])
    fig.colorbar(im, ax=ax, label='Coherence')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Frequency (per year)')
    ax.set_title('Wavelet Coherence')
    fig.tight_layout()
    return fig
    return fig
    return fig
