import pandas as pd

df = pd.read_csv(
    "../household_power_consumption.txt",
    sep=";",
    na_values=["?"],
    low_memory=False
)

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True
)

df = df.drop(columns=["Date", "Time"])
df = df.dropna()
df = df.sort_values("Datetime")

df.to_csv("../cleaned_energy.csv", index=False)

print("Data cleaning completed!")
print("Shape:", df.shape)
print(df.head())