import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../cleaned_energy.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])

print(df.head())
print()
print(df.describe())

plt.figure(figsize=(12, 5))
plt.plot(df["Datetime"], df["Global_active_power"])
plt.xlabel("Date")
plt.ylabel("Power Consumption")
plt.title("Household Energy Consumption")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()