from flask import Flask, render_template, request
import datetime
import numpy as np
from utils.wavelet_utils import pull_data, workflow

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    stocks_close_img = None

    # Move these inside the function so they update on every request
    now = datetime.datetime.now().replace(second=0, microsecond=0)
    default_end = now
    default_start = now - datetime.timedelta(days=5)
    default_samples = 1000

    ticker1 = request.form.get('ticker1', 'BTCUSDT')
    ticker2 = request.form.get('ticker2', 'ETHUSDT')
    start_date = request.form.get('start_date') or default_start.strftime('%Y-%m-%dT%H:%M')
    end_date = request.form.get('end_date') or datetime.datetime.now().replace(second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M')
    num_samples = request.form.get('num_samples') or str(default_samples)

    if request.method == 'POST':
        # Parse and validate form input
        try:
            num_samples_int = int(num_samples)
            if num_samples_int < 2 or num_samples_int > 1000:
                raise ValueError("Number of samples must be between 2 and 1000 (Binance API limit).")
        except Exception:
            error = "Invalid number of samples."
            return render_template('index.html', error=error, stocks_close_img=stocks_close_img,
                                   ticker1=ticker1, ticker2=ticker2,
                                   start_date=start_date, end_date=end_date, num_samples=num_samples)

        try:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%dT%H:%M")
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%dT%H:%M")
        except Exception:
            error = "Invalid date/time format."
            return render_template('index.html', error=error, stocks_close_img=stocks_close_img,
                                   ticker1=ticker1, ticker2=ticker2,
                                   start_date=start_date, end_date=end_date, num_samples=num_samples)

        now_dt = datetime.datetime.now()
        if end_dt <= start_dt:
            error = "End date/time must be after start date/time."
            return render_template('index.html', error=error, stocks_close_img=stocks_close_img,
                                   ticker1=ticker1, ticker2=ticker2,
                                   start_date=start_date, end_date=end_date, num_samples=num_samples)
        if start_dt > now_dt or end_dt > now_dt:
            error = "Start and end date/time must not be in the future."
            return render_template('index.html', error=error, stocks_close_img=stocks_close_img,
                                   ticker1=ticker1, ticker2=ticker2,
                                   start_date=start_date, end_date=end_date, num_samples=num_samples)

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        try:
            # Pull all available data in the range (up to 1000 points per Binance API)
            times1, close1, detrended1, _ = pull_data(ticker1, start_ms=start_ms, end_ms=end_ms, num_samples=1000)
            times2, close2, detrended2, _ = pull_data(ticker2, start_ms=start_ms, end_ms=end_ms, num_samples=1000)

            # Check for empty data after pulling
            if len(times1) == 0 or len(times2) == 0:
                error = "No data available for the selected date/time range. Please choose a different range."
                return render_template('index.html', error=error, stocks_close_img=stocks_close_img,
                                       ticker1=ticker1, ticker2=ticker2,
                                       start_date=start_date, end_date=end_date, num_samples=num_samples)

            # Use detrended for analysis, but keep close for plotting
            sig1 = detrended1
            sig2 = detrended2

            # Interpolate to num_samples evenly spaced points between start and end
            def interp_to_n(arr, times, n, start_ms, end_ms):
                # Ensure times and arr are sorted and unique
                sort_idx = np.argsort(times)
                times_sorted = times[sort_idx]
                arr_sorted = arr[sort_idx]
                # Remove duplicate times (keep last occurrence)
                _, unique_idx = np.unique(times_sorted, return_index=True)
                times_unique = times_sorted[unique_idx]
                arr_unique = arr_sorted[unique_idx]
                # Only interpolate if we have at least two unique points and n <= len(arr_unique)
                if len(arr_unique) < 2 or n > len(arr_unique):
                    # If not enough data, just return what we have (avoid out of bounds)
                    return arr_unique, times_unique
                new_times = np.linspace(times_unique[0], times_unique[-1], n)
                arr_interp = np.interp(new_times, times_unique, arr_unique)
                return arr_interp, new_times.astype(np.int64)

            sig1, times1 = interp_to_n(sig1, times1, num_samples_int, start_ms, end_ms)
            sig2, times2 = interp_to_n(sig2, times2, num_samples_int, start_ms, end_ms)
            close1, _ = interp_to_n(close1, times1, num_samples_int, start_ms, end_ms)
            close2, _ = interp_to_n(close2, times2, num_samples_int, start_ms, end_ms)

            # Run workflow
            max_period = 125
            min_period = 8
            bounds = (max_period, min_period)
            phase = True
            _, _, _, coherence_fig, stock_fig, stock_close_fig = workflow(
                sig1, sig2, bounds, phase=phase,
                ticker1=ticker1, ticker2=ticker2, close1=close1, close2=close2, times1=times1, times2=times2,
                sampling_rate=num_samples_int
            )
            coherence_fig.savefig('static/output.png')
            stock_fig.savefig('static/stocks.png')
            if stock_close_fig is not None:
                stock_close_fig.savefig('static/stocks_close.png')
                stocks_close_img = 'stocks_close.png'
        except Exception as e:
            error = str(e)

    return render_template(
        'index.html',
        error=error,
        stocks_close_img=stocks_close_img,
        ticker1=ticker1,
        ticker2=ticker2,
        start_date=start_date,
        end_date=end_date,
        num_samples=num_samples
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)