"""Re-apply the current answer parser to saved results (no model calls) and report what changed."""
import sys

import pandas as pd

from evaluate import extract_answer, summarize

for path in sys.argv[1:]:
    results = pd.read_csv(path)
    new_prediction = [extract_answer(r, p) for r, p in zip(results["raw_response"], results["prompt"])]
    old_prediction = results["prediction"].where(results["prediction"].notna(), None)

    changed = results[[old != new for old, new in zip(old_prediction, new_prediction)]]
    print(f"{path}: {len(changed)} of {len(results)} predictions changed")
    for index, row in changed.iterrows():
        print(f"  {row['language']} id {row['id']}: {row['prediction']} -> {new_prediction[index]} (gold {row['gold']})")

    results["prediction"] = new_prediction
    results["correct"] = results["prediction"] == results["gold"]
    results.to_csv(path, index=False)

    print("Unparsed responses:", results["prediction"].isna().sum())
    print(summarize(results).to_string(), "\n")
