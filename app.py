from flask import request, jsonify
from utils.coherence_utils import coherence, transform
from scipy.signal import detrend
import numpy as np

@app.route('/api/detrend', methods=['POST'])
def detrend_data():
    data = request.json.get('data')
    if not data:
        return jsonify([])
    return jsonify(detrend(data).tolist())

@app.route('/api/coherence', methods=['POST'])
def calculate_coherence():
    btc = request.json.get('btc', [])
    eth = request.json.get('eth', [])
    frame_rate = request.json.get('frame_rate', 1)
    
    if not (btc and eth):
        return jsonify({})
        
    # Transform signals
    highest_freq = 1/5   # 5 second minimum period
    lowest_freq = 1/40   # 40 second maximum period
    
    coeffs1, freqs = transform(btc, frame_rate, highest_freq, lowest_freq)
    coeffs2, _ = transform(eth, frame_rate, highest_freq, lowest_freq)
    
    # Calculate coherence
    coh, freqs, cross = coherence(coeffs1, coeffs2, freqs)
    
    # Convert frequencies to periods
    periods = 1/freqs
    
    return jsonify({
        'coherence': coh.tolist(),
        'periods': periods.tolist(),
        'times': list(range(len(btc))),
        'phase': np.angle(cross).tolist()
    })