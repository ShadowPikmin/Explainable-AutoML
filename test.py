import pandas as pd

df = pd.read_csv("tpot_small_results.csv")
print(df.columns.tolist())