import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import cvxpy as cp

# ====== 設定 ======
start = datetime.datetime(2005, 1, 1)
end = datetime.datetime(2024, 12, 31)
treatment_date = pd.Timestamp('2016-06-01')  # Brexit投票時期

# FREDコード（実質GDP系列）
codes = {
    'United Kingdom': 'CLVMNACSCAB1GQUK',
    'Germany': 'CLVMNACSCAB1GQDE',
    'France': 'CLVMNACSCAB1GQFR',
    'Italy': 'CLVMNACSCAB1GQIT',
    'Spain': 'CLVMNACSCAB1GQES'
}

# ====== データ取得 ======
df = pd.DataFrame()
for country, code in codes.items():
    try:
        data = web.DataReader(code, 'fred', start, end)
        df[country] = data[code]
        print(f"✅ Loaded: {country}")
    except Exception as e:
        print(f"⚠️ Error loading {country}: {e}")

# 欠損値処理・2015=100基準化
df = df.interpolate(limit_direction='both')
nearest = df.index.get_indexer([pd.Timestamp('2015-01-01')], method='nearest')[0]
df = df / df.iloc[nearest] * 100

# ====== SCM設定 ======
treated = 'United Kingdom'
donors = ['Germany', 'France', 'Italy', 'Spain']

# プレ処置期間を直前x年（前4x四半期）に限定
pre_period = df.loc[df.index < treatment_date].iloc[-40:]
post_period = df.loc[df.index >= treatment_date]

# ====== SCM推定 ======
X1 = pre_period[treated].values
X0 = pre_period[donors].values

w = cp.Variable(len(donors), nonneg=True)
constraints = [cp.sum(w) == 1]
objective = cp.Minimize(cp.sum_squares(X1 - X0 @ w))
problem = cp.Problem(objective, constraints)
problem.solve()

weights = w.value
print("\n=== 最適ドナー国重み ===")
for d, val in zip(donors, weights):
    print(f"{d}: {val:.3f}")

# ====== 可視化用データ範囲 ======
df_plot = pd.concat([df.loc[df.index < treatment_date], df.loc[df.index >= treatment_date]])
df_plot = df_plot.copy()
df_plot['Synthetic UK'] = df_plot[donors].values @ weights

# ====== グラフ描画 (表題年数変えるの忘れずに)======
plt.figure(figsize=(10,6))
plt.plot(df_plot.index, df_plot[treated], label='United Kingdom (Actual)', linewidth=2)
plt.plot(df_plot.index, df_plot['Synthetic UK'], label='Synthetic United Kingdom', linestyle='--', linewidth=2)
plt.axvline(treatment_date, color='gray', linestyle=':', label='Brexit vote (2016Q2)')
plt.title('Synthetic Control for UK Real GDP (Pre-treatment: 10 years)')
plt.xlabel('Year')
plt.ylabel('Real GDP (2015=100)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
