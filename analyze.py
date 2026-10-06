"""Collect every mistake from the subset runs and summarize the error patterns."""
import json
import re

import pandas as pd

from evaluate import LANGUAGES, parse_options

RUNS = {
    "Gemini 3.8 Flash": "results/gemini-3.8-flash_subset.csv",
    "Llama 3 8B": "results/llama3_8b-instruct-q4_K_M_subset.csv",
}
OUTPUT = "results/mistakes_subset.csv"
JSON_OUTPUT = "results/mistakes_subset.json"

questions = pd.read_csv("test_suite.csv").set_index("id")


def utterance(prompt):
    match = re.search(r'^["“](.+?)["”]\s*$', prompt, re.MULTILINE)
    return match.group(1) if match else ""


def error_kind(gold, prediction):
    if prediction is None or pd.isna(prediction):
        return "no answer found"
    if gold == "E":
        return "missed 'None of the above'"
    if prediction == "E":
        return "wrongly chose 'None of the above'"
    return "chose a wrong option"


frames = []
for model, path in RUNS.items():
    run = pd.read_csv(path)
    run["model"] = model
    frames.append(run)
results = pd.concat(frames, ignore_index=True)

mistakes = results[~results["correct"]].copy()
english_options = {i: parse_options(questions.loc[i, "english"]) for i in mistakes["id"].unique()}
mistakes["utterance_english"] = [utterance(questions.loc[i, "english"]) for i in mistakes["id"]]
mistakes["gold_text_english"] = [english_options[i][g] for i, g in zip(mistakes["id"], mistakes["gold"])]
mistakes["chosen_text_english"] = [
    english_options[i].get(p, "") if isinstance(p, str) else ""
    for i, p in zip(mistakes["id"], mistakes["prediction"])
]
mistakes["error_kind"] = [error_kind(g, p) for g, p in zip(mistakes["gold"], mistakes["prediction"])]
languages_wrong = mistakes.groupby(["model", "id"])["language"].nunique().rename("languages_wrong")
mistakes = mistakes.join(languages_wrong, on=["model", "id"])

columns = ["model", "language", "id", "type", "gold", "prediction", "error_kind", "languages_wrong",
           "utterance_english", "gold_text_english", "chosen_text_english", "raw_response"]
mistakes = mistakes[columns].sort_values(["model", "id", "language"])
mistakes.to_csv(OUTPUT, index=False)

# Compact version of the same data for the interactive mistakes view.
letter_share = results.groupby("model")["prediction"].value_counts(normalize=True).unstack().fillna(0) * 100
compact = {
    "mistakes": [
        {
            "m": r.model, "lang": r.language, "id": int(r.id), "type": r.type, "gold": r.gold,
            "pred": r.prediction if isinstance(r.prediction, str) else None,
            "kind": r.error_kind, "nLang": int(r.languages_wrong),
            "excerpt": " ".join(str(r.raw_response).split())[:170],
        }
        for r in mistakes.itertuples()
    ],
    "questions": {
        str(i): {
            "utterance": utterance(questions.loc[i, "english"]),
            "options": {
                letter: text for letter, text in english_options[i].items()
                if letter in set(mistakes.loc[mistakes["id"] == i, "gold"])
                | set(mistakes.loc[mistakes["id"] == i, "prediction"].dropna())
            },
        }
        for i in sorted(mistakes["id"].unique())
    },
    "letterShare": {m: {k: round(v, 1) for k, v in row.items()} for m, row in letter_share.iterrows()},
}
with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(compact, f, ensure_ascii=False, separators=(",", ":"))

if __name__ == "__main__":
    print(f"{len(mistakes)} mistakes saved to {OUTPUT}\n")

    print("Mistakes per model and language (out of 90 questions):")
    print(pd.crosstab(mistakes["model"], mistakes["language"]).reindex(columns=LANGUAGES).to_string(), "\n")

    print("Mistakes per model and category (out of 18 x 4 = 72 per category):")
    print(pd.crosstab(mistakes["model"], mistakes["type"]).to_string(), "\n")

    print("Kind of mistake:")
    print(pd.crosstab(mistakes["error_kind"], mistakes["model"]).to_string(), "\n")

    print("How many languages each missed question was missed in:")
    per_question = mistakes.drop_duplicates(["model", "id"])
    print(pd.crosstab(per_question["languages_wrong"], per_question["model"]).to_string(), "\n")

    print("Letter chosen, all answers (gold letters are 20% each):")
    share = results.groupby("model")["prediction"].value_counts(normalize=True).unstack().fillna(0) * 100
    print(share.round(1).to_string(), "\n")

    wrong_ids = {m: set(zip(g["language"], g["id"])) for m, g in mistakes.groupby("model")}
    both = wrong_ids["Gemini 3.8 Flash"] & wrong_ids["Llama 3 8B"]
    print(f"Question/language pairs both models got wrong: {len(both)} "
          f"(Gemini has {len(wrong_ids['Gemini 3.8 Flash'])} mistakes in total)")
