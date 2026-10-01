import pandas as pd

df = pd.read_csv(
    "household_power_consumption.txt",
    sep=";",
    na_values=["?"],
    low_memory=False
)

print(df.head())
print("Shape:", df.shape)
print(df.info())