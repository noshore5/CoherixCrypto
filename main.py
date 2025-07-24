from fastapi import FastAPI, Request, Query, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, conlist
from typing import List
import numpy as np
import uvicorn
from scipy.signal import detrend
from matplotlib import pyplot as plt

from utils.binance_utils import get_binance_live_data  # or get_live_data, choose one
from utils.coherence_utils import coherence, transform

app = FastAPI()

# Static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# HTML route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("coherixcrypto.html", {"request": request})

# API route for live data
@app.get("/api/live-data")
async def live_data(range: str = Query("60s")):
    # Pass the range to the data fetcher
    data = await get_binance_live_data(range=range)  # use the function you've imported
    return data

class DetrendRequest(BaseModel):
    btc: conlist(float, min_length=2)  # Require at least 2 points
    eth: conlist(float, min_length=2)

@app.post("/api/detrend")
async def detrend_data(data: DetrendRequest):
    try:
        if len(data.btc) != len(data.eth):
            raise HTTPException(
                status_code=422,
                detail="BTC and ETH arrays must have same length"
            )
        
        btc_detrended = detrend(np.array(data.btc, dtype=np.float64)).tolist()
        eth_detrended = detrend(np.array(data.eth, dtype=np.float64)).tolist()
        
        return {
            "btc_detrended": btc_detrended,
            "eth_detrended": eth_detrended
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/coherence")
async def calculate_coherence(data: dict = Body(...)):
    try:
        btc = data.get("btc")
        eth = data.get("eth")
        if not isinstance(btc, list) or not isinstance(eth, list):
            raise HTTPException(status_code=422, detail="btc and eth must be lists")
        if len(btc) < 2 or len(eth) < 2:
            raise HTTPException(status_code=422, detail="Insufficient data points")
        if len(btc) != len(eth):
            raise HTTPException(status_code=422, detail="BTC and ETH arrays must have same length")

        btc_arr = np.array([float(x) for x in btc], dtype=np.float64)
        eth_arr = np.array([float(x) for x in eth], dtype=np.float64)
        frame_rate = 1.0
        highest_freq = 1/40    # lowest period = 40s
        lowest_freq = 1/500    # largest period = 500s
        nfreqs = 100

        coeffs1, freqs = transform(btc_arr, frame_rate, highest_freq, lowest_freq, nfreqs=nfreqs)
        coeffs2, _ = transform(eth_arr, frame_rate, highest_freq, lowest_freq, nfreqs=nfreqs)
        coh, freqs, cross = coherence(coeffs1, coeffs2, freqs)
        periods = 1 / freqs

        # Normalize coherence array before plotting
        coh = np.abs(coh)
        if np.max(coh) > 0:
            coh = coh / np.max(coh)

        coherence_list = coh.tolist()
        periods_list = np.array(periods).tolist()
        phase_list = np.angle(np.array(cross)).tolist()
        time_list = list(range(np.array(coh).shape[1]))

        return {
            "coherence": coherence_list,
            "periods": periods_list,
            "time": time_list,
            "phase": phase_list
        }
    except Exception as e:
        print(f"Error calculating coherence: {e}")  # Debugging line
        raise HTTPException(status_code=422, detail=str(e))



# Uvicorn entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
