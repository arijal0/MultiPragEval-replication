import pandas as pd

df = pd.read_csv("test_suite.csv")

print("Number of rows:", len(df))
print()
print("Columns:")
print(df.columns.tolist())

print("\nTypes:")
print(df["type"].value_counts())

print("\nGold-answer distribution:")
print(df["answer"].value_counts())

print("\nFirst row:")
print(df.iloc[0])
