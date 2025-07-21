from flask import Flask, render_template, request

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.wavelet_utils2 import pull_data, workflow, API_KEY
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    stocks_close_img = None
    if request.method == 'POST':
        ticker1 = request.form['ticker1'].upper()
        ticker2 = request.form['ticker2'].upper()
        try:
            max_period = 125
            min_period = 8
            phase = True
            times1, close1, sig1, _ = pull_data(ticker1, API_KEY)
            times2, close2, sig2, _ = pull_data(ticker2, API_KEY)
            _, _, _, coherence_fig, stock_fig, stock_close_fig = workflow(
                sig1, sig2, (max_period, min_period), phase=phase,
                ticker1=ticker1, ticker2=ticker2, close1=close1, close2=close2, times1=times1, times2=times2
            )
            coherence_fig.savefig('static/output.png')
            stock_fig.savefig('static/stocks.png')
            if stock_close_fig is not None:
                stock_close_fig.savefig('static/stocks_close.png')
                stocks_close_img = 'stocks_close.png'
            result = 'output.png'
        except Exception as e:
            error = str(e)
    return render_template('index.html', result=result, error=error, stocks_close_img=stocks_close_img)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)