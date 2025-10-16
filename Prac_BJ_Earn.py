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

# 四半期データ
decomposition = seasonal_decompose(df['Earn'], model='additive', period=4)

# プロット設定
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# 結果の描画 (一旦コメントアウト)
#decomposition.plot()
#plt.show()

#Determing rolling statistics
rolmean = pd.Series(df['Earn']).rolling(window=12).mean()
rolstd = pd.Series(df['Earn']).rolling(window=12).std()

#Plot rolling statistics:
#orig = plt.plot(df['Earn'], color='blue',label='Original')
#mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#std = plt.plot(rolstd, color='black', label = 'Rolling Std')
#plt.legend(loc='best')
#plt.title('Rolling Mean & Standard Deviation')
#plt.xlabel('Timeline')
#plt.ylabel('Earnings (pound)')
#plt.rcParams['figure.facecolor'] = 'white'
#plt.rcParams['axes.facecolor'] = 'white'
#plt.show()

#Dicky-Fuller Test
#print ('Results of Dickey-Fuller Test:')
dftest = adfuller(df['Earn'])

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
#print(dfoutput)

#ACF(Autocorrelation function)のプロット
from statsmodels.graphics.tsaplots import plot_acf
#plt.rcParams.update({'figure.figsize':(8,6), 'figure.dpi':120})
#plot_acf(df['Earn'])
#plt.show()

#PACF(Partial Autocorrelation function)のプロット
from statsmodels.graphics.tsaplots import plot_pacf
#plot_pacf(df['Earn'])
#plt.show()

# ADF Test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(); print(f'ADF Statistic: {result[0]}')
    print();  print(f'n_lags: {result[1]}')
    print();  print(f'p-value: {result[1]}')

    print(); print('Critial Values:')
    for key, value in result[4].items():
        print(f'   {key}, {value}')   

#adf_test(df["Earn"])

# KPSS Test
from statsmodels.tsa.stattools import kpss

def kpss_test(series, **kw):    
    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    
    # Format Output
    print(); print(f'KPSS Statistic: {statistic}')
    print(); print(f'p-value: {p_value}')
    print(); print(f'num lags: {n_lags}')
    print(); print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    
#kpss_test(df["Earn"])

#ARIMAモデルの適用
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df["Earn"], order=(0,1,1))
model_fit = model.fit()
#print(model_fit.summary())

# Residuals
residuals = pd.DataFrame(model_fit.resid)
#plt.rcParams.update({'figure.figsize':(12,5), 'figure.dpi':120})
fig, ax = plt.subplots(1, 2)
# Residuals 時系列プロット（縦軸は-100～100）
residuals.plot(title="Residuals", ax=ax[0])
ax[0].set_ylim(-100, 100)
# Densityプロット（横軸は-100～100）
residuals.plot(kind='kde', title='Density', ax=ax[1])
ax[1].set_xlim(-100, 100)
#plt.tight_layout()
#plt.show()

# 2012年以降に絞る
start_date = '2012-01-01'
actual_post2012 = df['Earn'][start_date:]
fitted_post2012 = model_fit.fittedvalues[start_date:]
plt.figure(figsize=(12,5))
plt.plot(actual_post2012, label='Actual')
plt.plot(fitted_post2012, label='Fitted', linestyle='--')
plt.title('Actual vs Fitted (2013 onwards)')
plt.xlabel('Timeline')
plt.ylabel('Earnings (pound)')
# 縦軸は0付近を見やすく
min_val = min(actual_post2012.min(), fitted_post2012.min())
max_val = max(actual_post2012.max(), fitted_post2012.max())
plt.ylim(min_val*0.95, max_val*1.05)
#plt.legend()
#plt.show()

#Forecast
# 予測範囲を指定
forecast_start = '2021-06-01'
forecast_end = '2026-12-01'

# 予測を取得
forecast = model_fit.get_forecast(steps=20)  # 未来20期間など適宜変更
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int()

# プロット設定
fig, ax = plt.subplots(figsize=(12, 6))
df['Earn'].loc['2011-03-01':].plot(ax=ax, label='Observed')
pred_mean.plot(ax=ax, label='Forecast', color='r')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)

plt.title('Forecast', fontsize=40)
plt.xlabel('Timeline', fontsize=25)
plt.ylabel('Earnings (pound)', fontsize=25)
plt.legend(fontsize=20)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.show()