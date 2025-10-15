import datetime
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web

# 期間設定
start = datetime.datetime(1990, 1, 1)
end = datetime.datetime(2024, 12, 31)

# 各国のFREDコード
codes = {
    'United Kingdom': 'CLVMNACSCAB1GQUK',  # 実質GDP（2015=100）
    'United States': 'GDPC1',              # 実質GDP（10億ドル）
    'Germany': 'CLVMNACSCAB1GQDE',         # 実質GDP（2015=100）
    'Japan': 'JPNRGDPEXP'                  # 実質GDP（実額, 代替コード）
}

# データ取得
df = pd.DataFrame()
for country, code in codes.items():
    try:
        data = web.DataReader(code, 'fred', start, end)
        df[country] = data[code]
        print(f"✅ Loaded: {country} ({len(data)} observations)")
    except Exception as e:
        print(f"⚠️ Error loading {country}: {e}")

# 欠損を補間
df = df.interpolate()

# 2015年を基準に指数化
nearest = df.index.get_indexer([pd.Timestamp('2015-01-01')], method='nearest')[0]
df = df / df.iloc[nearest] * 100

# プロット
plt.figure(figsize=(10, 6))
for country in df.columns:
    plt.plot(df.index, df[country], label=country)

plt.title('Real GDP (Indexed to 2015 = 100, Quarterly, FRED)')
plt.xlabel('Year')
plt.ylabel('Index (2015 = 100)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
