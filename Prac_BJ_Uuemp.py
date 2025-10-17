import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

# Excelファイルの読み込み
df = pd.read_excel("Box-Jenkins.xlsx", sheet_name="Sheet2")

# 整形 & PeriodIndex で変換
df['date'] = pd.PeriodIndex(df['Year_Quarter'].str.replace(' ', ''), freq='Q').to_timestamp()

# インデックスに設定
df.set_index('date', inplace=True)

# 四半期データの分解（失業率）
decomposition = seasonal_decompose(df['Unemp'], model='additive', period=4)

# プロット設定
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# 結果の描画 
decomposition.plot()
plt.show()

# ローリング平均と標準偏差の計算
rolmean = pd.Series(df['Unemp']).rolling(window=12).mean()
rolstd = pd.Series(df['Unemp']).rolling(window=12).std()

# プロット
orig = plt.plot(df['Unemp'], color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation (Unemployment)')
plt.xlabel('Timeline')
plt.ylabel('Unemployment Rate')
plt.show()

# Dickey-Fuller Test
print('Results of Dickey-Fuller Test:')
dftest = adfuller(df['Unemp'])
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)

# ACFのプロット
plt.rcParams.update({'figure.figsize': (8, 6), 'figure.dpi': 120})
plot_acf(df['Unemp'])
plt.show()

# PACFのプロット
plot_pacf(df['Unemp'])
plt.show()

# ADF Test 関数
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(); print(f'ADF Statistic: {result[0]}')
    print(); print(f'p-value: {result[1]}')
    print(); print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}, {value}')   

adf_test(df["Unemp"])

# KPSS Test 関数
from statsmodels.tsa.stattools import kpss

def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    print(); print(f'KPSS Statistic: {statistic}')
    print(); print(f'p-value: {p_value}')
    print(); print(f'num lags: {n_lags}')
    print(); print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')    

kpss_test(df["Unemp"])

# ARIMAモデルの適用
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df["Unemp"], order=(0,1,1))
model_fit = model.fit()
print(model_fit.summary())

# Residuals
residuals = pd.DataFrame(model_fit.resid)
plt.rcParams.update({'figure.figsize': (12,5), 'figure.dpi': 120})
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
ax[0].set_ylim(-5, 5)
residuals.plot(kind='kde', title='Density', ax=ax[1])
ax[1].set_xlim(-5, 5)
plt.tight_layout()
plt.show()

# 2012年以降に絞る
start_date = '2012-01-01'
actual_post2012 = df['Unemp'][start_date:]
fitted_post2012 = model_fit.fittedvalues[start_date:]
plt.figure(figsize=(12,5))
plt.plot(actual_post2012, label='Actual')
plt.plot(fitted_post2012, label='Fitted', linestyle='--')
plt.title('Actual vs Fitted (2013 onwards) - Unemployment')
plt.xlabel('Timeline')
plt.ylabel('Unemployment Rate')
min_val = min(actual_post2012.min(), fitted_post2012.min())
max_val = max(actual_post2012.max(), fitted_post2012.max())
plt.ylim(min_val*0.95, max_val*1.05)
plt.legend()
plt.show()

# Forecast
import pmdarima as pm
model = pm.auto_arima(df['Unemp'], seasonal=False, trend='t', d=1)

forecast = model.predict(n_periods=20, return_conf_int=True)
forecast_index = pd.date_range(df.index[-1], periods=20, freq='Q')

plt.figure(figsize=(12, 6))
plt.plot(df['Unemp'], label='Observed')
plt.plot(forecast_index, forecast[0], label='Forecast', color='r')
plt.fill_between(forecast_index, forecast[1][:, 0], forecast[1][:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title("Forecast of Unemployment with Trend Included")
plt.show()
