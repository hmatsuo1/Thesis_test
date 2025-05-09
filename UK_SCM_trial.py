import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

# FRED APIキーを設定
API_KEY = '880394291952f6c36a093c64e22a9527'
fred = Fred(api_key=API_KEY)

# 各国のシリーズID
countries = {
    'UK': 'NGDPRSAXDCGBQ',
    'Germany': 'CLVMNACSCAB1GQDE',
    'France': 'CLVMNACSCAB1GQFR',
    'Italy': 'CLVMNACSCAB1GQIT',
    'Canada': 'NGDPRSAXDCCAQ',
    'Japan': 'JPNRGDPEXP',
    'USA': 'GDPC1'
}

# データを取得し整形
gdp_data = pd.DataFrame()
for country, code in countries.items():
    try:
        data = fred.get_series(code)
        gdp_data[country] = data
    except Exception as e:
        print(f"Error retrieving data for {country}: {e}")

# インデックスを四半期に変換
gdp_data.index = pd.to_datetime(gdp_data.index)
gdp_data = gdp_data.dropna()
# 四半期で整形（推奨形式）
gdp_data = gdp_data.resample('Q-DEC').mean()

# 欠損のある列は削除
gdp_data = gdp_data.dropna(axis=1)

# 基準年が存在するか確認してインデックス化
base_period = '2015-01-01'
if base_period not in gdp_data.index:
    raise ValueError(f"Base period {base_period} not found in GDP data.")
gdp_indexed = gdp_data / gdp_data.loc[base_period] * 100

# SCM用のデータ整備
treated = 'UK'
donors = [c for c in gdp_indexed.columns if c != treated]

# プレトリートメント期間定義
pre_period = (gdp_indexed.index >= '2010-01-01') & (gdp_indexed.index <= '2019-12-31')
post_period = (gdp_indexed.index > '2019-12-31')

X_pre = gdp_indexed.loc[pre_period, donors].values
y_pre = gdp_indexed.loc[pre_period, treated].values

# エラーチェック
if X_pre.shape[0] != y_pre.shape[0]:
    raise ValueError("X_pre and y_pre have mismatched lengths.")

# 線形回帰によるSCM
reg = LinearRegression(fit_intercept=False)
reg.fit(X_pre, y_pre)
weights = reg.coef_

# SCM予測
synthetic = gdp_indexed[donors].dot(weights)

# 結果プロット
plt.figure(figsize=(10,6))
plt.plot(gdp_data.index, gdp_data[treated], label='UK (Actual)', color='black')
plt.plot(gdp_data.index, synthetic, label='Synthetic UK', color='red', linestyle='--')
plt.axvline(pd.to_datetime(intervention_date), color='gray', linestyle=':', label='COVID Start')
plt.title('SCM: Impact of COVID-19 on UK Real GDP')
plt.ylabel('Real GDP Index')
plt.legend()
plt.grid(True)
plt.show()


