# 各国の実質GDP推移を同一グラフ上で比較表示するコード

from datetime import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt

# 分析期間
start = datetime(1990, 1, 1)
end = datetime(2025, 1, 1)

# OECD基準の実質GDP系列（2015=100、季節調整済指数）
# 各国で整合性が取れるように OECD の “MEI” シリーズを使用
countries = {
    'United Kingdom': 'GDPGBPQDSMEI',
    'United States': 'GDPUSQDSMEI',
    'Germany': 'GDPDEUQDSMEI',
    'Japan': 'GDPJPQDSMEI'
}

# データ格納用
gdp_data = {}

# 各国のGDPデータを取得
for country, code in countries.items():
    try:
        data = web.DataReader(code, 'fred', start, end)
        gdp_data[country] = data
        print(f"{country} data retrieved successfully ({len(data)} observations)")
    except Exception as e:
        print(f"Error retrieving {country}: {e}")



for country, data in gdp_data.items():
    print(f"{country}: {data.shape}")
