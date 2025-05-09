import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from sklearn.linear_model import LinearRegression

# FRED APIキーをここに入力してください
API_KEY = '880394291952f6c36a093c64e22a9527'
fred = Fred(api_key=API_KEY)

# 各国の実質GDPシリーズID（四半期データ）
countries = {
    'UK': 'NGDPRSAXDCGBQ',
    'Germany': 'CLVMNACSCAB1GQDE',
    'France': 'CLVMNACSCAB1GQFR',
    'Italy': 'CLVMNACSCAB1GQIT',
    'Canada': 'NGDPRSAXDCCAQ',
    'Japan': 'JPNRGDPEXP',
    'USA': 'GDPC1'
}

# データ取得と整形
gdp_data = pd.DataFrame()
for country, code in countries.items():
    try:
        series = fred.get_series(code)
        gdp_data[country] = series
    except Exception as e:
        print(f"Error loading {country}: {e}")

# 時系列の統一・欠損処理
gdp_data.index = pd.to_datetime(gdp_data.index)
gdp_data = gdp_data.resample('QE-DEC').mean()  # 四半期ベースで整形（推奨形式）
gdp_data = gdp_data.dropna(axis=1)  # 欠損のある列を削除

# 基準年（2015年Q1）でインデックス化（2015Q1 = 100）
base_period = '2015-03-31'
if base_period not in gdp_data.index:
    print("利用可能なインデックス一覧:", gdp_data.index)
    raise ValueError(f"Base period {base_period} not found in GDP data.")
gdp_indexed = gdp_data / gdp_data.loc[base_period] * 100

# SCMの準備
treated = 'UK'
donors = [c for c in gdp_indexed.columns if c != treated]

# プレトリートメント期間：2010年Q1～2019年Q4
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

# プロット範囲（2010年Q1以降）
plot_index = gdp_indexed.index >= '2010-01-01'

# 実GDP vs 合成GDP
plt.figure(figsize=(10, 6))
plt.plot(gdp_indexed.index[plot_index], gdp_indexed[treated][plot_index], label='UK (Actual)', linewidth=2)
plt.plot(gdp_indexed.index[plot_index], synthetic[plot_index], label='Synthetic UK', linestyle='--')
plt.axvline(pd.to_datetime('2020-01-01'), color='gray', linestyle=':', label='COVID Shock')
plt.title('Synthetic Control Method: Real GDP (Indexed to 2015Q1 = 100)')
plt.ylabel('Real GDP Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# SCMギャップ（差分）のプロット
plt.figure(figsize=(10, 4))
plt.plot(gdp_indexed.index[plot_index], scm_gap[plot_index], label='Gap (UK - Synthetic)', color='red')
plt.axvline(pd.to_datetime('2020-01-01'), color='gray', linestyle=':')
plt.title('SCM Gap: Impact of COVID on UK Real GDP')
plt.ylabel('Gap in Index Points')
plt.grid(True)
plt.tight_layout()
plt.show()
