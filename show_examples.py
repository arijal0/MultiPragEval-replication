import pandas as pd

df = pd.read_csv("test_suite.csv")

for category in df["type"].unique():
    print("\n" + "=" * 60)
    print(category.upper())
    print("=" * 60)

    category_rows = df[df["type"] == category]
    # Rows are sorted by gold answer in blocks of 12 (A..E), so step by 12 to get one of each.
    sample = category_rows.iloc[::12]

    for _, row in sample.iterrows():
        print("\nID:", row["id"])
        print(row["english"])
        print("GOLD:", row["answer"])
