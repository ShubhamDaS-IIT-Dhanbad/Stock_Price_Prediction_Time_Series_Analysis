# Tesla Stock Price Time Series Analysis 📈

## Overview  
This project performs **time series analysis and forecasting** on Tesla stock prices using classical statistical models such as **AR, MA, ARMA, ARIMA, VAR, VMA, VARMA, and VARIMA**.  

We apply **stationarity tests** (ADF, KPSS, KS) to check whether the data is stationary, use **differencing** where required, and fit multiple models to forecast Tesla stock closing prices.

---

## Dataset  
The dataset contains Tesla stock historical data with the following relevant columns:
- `Date`
- `Close`
- `roll` (rolling mean added for smoothing)

---

## Steps Followed  

### 1. Data Preprocessing  
- Selected columns: `Date`, `Close`, `roll`  
- Converted `Date` to `datetime` format  
- Set `Date` as the index for time series analysis  
- Checked for null values  

### 2. Stationarity Tests  
We applied three statistical tests:  
- **ADF (Augmented Dickey-Fuller Test)**  
  - Null Hypothesis: Series is non-stationary  
- **KPSS (Kwiatkowski–Phillips–Schmidt–Shin Test)**  
  - Null Hypothesis: Series is stationary  
- **Kolmogorov-Smirnov (KS Test)**  
  - Tests distribution differences  

**Result:** The differenced series was **stationary** (ADF p-value ≈ 1e-09, KPSS p-value > 0.05).  

### 3. Differencing  
- First differencing applied:  
  ```python
  df['Close_diff'] = df['Close'].diff()
