"""Compare Gemini with thinking off and thinking on on the same 90-question subset."""
from math import comb

import pandas as pd

from evaluate import LANGUAGES, PRAGMATIC_TYPES, summarize

OFF_PATH = "results/gemini-3.8-flash_subset.csv"
ON_PATH = "results/gemini-3.8-flash-thinking_subset.csv"
# The thinking-off run predates per-call timing; this is its wall-clock time divided by its 360 calls.
OFF_SECONDS_PER_CALL = 618 / 360


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


off = pd.read_csv(OFF_PATH)
on = pd.read_csv(ON_PATH)
key = ["language", "id"]
paired = off.merge(on, on=key, suffixes=("_off", "_on"))
paired["type"] = paired["type_off"]

if __name__ == "__main__":
    print(f"Paired answers: {len(paired)} (thinking off: {len(off)}, thinking on: {len(on)})\n")

    off_table, on_table = summarize(off), summarize(on)
    print("Thinking OFF, accuracy (%):")
    print(off_table.to_string(), "\n")
    print("Thinking ON, accuracy (%):")
    print(on_table.to_string(), "\n")
    print("Change, ON minus OFF (percentage points):")
    print((on_table - off_table).round(1).to_string(), "\n")

    print("Overall accuracy on all 360 answers: "
          f"off {off['correct'].mean() * 100:.1f}%, on {on['correct'].mean() * 100:.1f}%")
    four_maxims = paired[paired["type"].isin(PRAGMATIC_TYPES)]
    print("4-maxim accuracy (288 answers): "
          f"off {four_maxims['correct_off'].mean() * 100:.1f}%, on {four_maxims['correct_on'].mean() * 100:.1f}%\n")

    fixed = paired[~paired["correct_off"] & paired["correct_on"]]
    broken = paired[paired["correct_off"] & ~paired["correct_on"]]
    both_wrong = paired[~paired["correct_off"] & ~paired["correct_on"]]
    print(f"Answers thinking fixed (off wrong, on right): {len(fixed)}")
    print(f"Answers thinking broke (off right, on wrong): {len(broken)}")
    print(f"Wrong both times: {len(both_wrong)}")
    print(f"Exact McNemar test, two-sided p = {mcnemar_exact(len(fixed), len(broken)):.3f}\n")

    flips = pd.DataFrame({
        "fixed": fixed.groupby("language").size(),
        "broken": broken.groupby("language").size(),
        "wrong both times": both_wrong.groupby("language").size(),
    }).reindex(LANGUAGES).fillna(0).astype(int)
    print("Flips per language:")
    print(flips.to_string(), "\n")

    columns = ["language", "id", "type", "gold_on", "prediction_off", "prediction_on"]
    for title, rows in [("Fixed by thinking", fixed), ("Broken by thinking", broken),
                        ("Wrong both times", both_wrong)]:
        print(f"{title}:")
        print(rows[columns].sort_values(["id", "language"]).to_string(index=False) if len(rows) else "  none", "\n")

    print("Cost and speed per call (thinking ON, measured):")
    usage = on.groupby("language")[["seconds", "input_tokens", "output_tokens", "thinking_tokens"]].mean()
    usage.loc["all"] = on[["seconds", "input_tokens", "output_tokens", "thinking_tokens"]].mean()
    print(usage.reindex(LANGUAGES + ["all"]).round(1).to_string(), "\n")

    # The thinking-off run did not record tokens, so estimate them from reply length using
    # the tokens-per-character ratio of the visible replies in the thinking-on run, per language.
    on["chars"] = on["raw_response"].astype(str).str.len()
    off["chars"] = off["raw_response"].astype(str).str.len()
    ratio = on.groupby("language")["output_tokens"].sum() / on.groupby("language")["chars"].sum()
    off["estimated_output_tokens"] = off["chars"] * off["language"].map(ratio)
    off_tokens = off["estimated_output_tokens"].mean()
    on_billed = (on["output_tokens"] + on["thinking_tokens"]).mean()
    print("Generated tokens per call (thinking tokens are billed as output):")
    print(f"  thinking off: about {off_tokens:.0f} (estimated from reply length)")
    print(f"  thinking on:  {on_billed:.0f} ({on['output_tokens'].mean():.0f} visible + "
          f"{on['thinking_tokens'].mean():.0f} thinking), about {on_billed / off_tokens:.1f}x more")
    print(f"Seconds per call: off about {OFF_SECONDS_PER_CALL:.1f} (wall clock / calls), "
          f"on {on['seconds'].mean():.1f} (median {on['seconds'].median():.1f}, max {on['seconds'].max():.1f})\n")

    print("Thinking tokens on answers that were right vs wrong (thinking ON):")
    print(on.groupby("correct")["thinking_tokens"].describe()[["count", "mean", "50%", "max"]].round(0).to_string())
