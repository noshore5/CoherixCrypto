from scipy.ndimage import gaussian_filter
from scipy.signal import detrend 
import fcwt  # Ensure you have the fcwt package installed
import numpy as np
import os

def coherence(coeffs1, coeffs2, freqs):
    S1 = np.abs(coeffs1) ** 2
    S2 = np.abs(coeffs2) ** 2
    S12 = coeffs1 * np.conj(coeffs2)
    def smooth(data, sigma=(.5, .5), mode='nearest'):
        return gaussian_filter(data, sigma=sigma, mode=mode)
    S1_smooth = smooth(S1)
    S2_smooth = smooth(S2)
    S12_smooth = smooth(np.abs(S12) ** 2)
    coh = S12_smooth / (np.sqrt(S1_smooth) * np.sqrt(S2_smooth))
    return coh, freqs, S12

def transform(signal1, frame_rate, highest, lowest, nfreqs=100):
    signal1 = np.asarray(signal1, dtype=np.float64)
    frame_rate = 1
    lowest = .002
    highest = .025
    nfreqs = 200

    freqs, coeffs1 = fcwt.cwt(signal1, frame_rate, lowest, highest, nfreqs, nthreads=4, scaling='log')
    return coeffs1, freqs
