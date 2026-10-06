import os
import re
import time
import unicodedata

import pandas as pd

LANGUAGES = ["english", "german", "korean", "chinese"]
PRAGMATIC_TYPES = ["quantity", "quality", "relation", "manner"]

# Korean models sometimes number options with circled digits instead of letters.
CIRCLED_NUMBERS = {"①": "(A)", "②": "(B)", "③": "(C)", "④": "(D)", "⑤": "(E)"}

# A capital option letter that is not the start of a word ("E" in "Emily" must not count).
LETTER = r"\(?([A-E])\)?(?![A-Za-z])"

# "Answer: B", "The answer is (A)", "Die Antwort ist (A)", "정답은 (C)", "답: (A)", "答案是D", "答:B",
# including Markdown bold such as "**Answer:** (A)" or "The answer is **(A)**".
# Conclusions such as "the correct choice is (E)" or "最恰当的选项是 (E)" need "is"/"是" right before
# the letter, so "Option (A) is incorrect" and "分析各选项：(A)" are not mistaken for answers.
ANSWER_LABEL = re.compile(
    r"(?:"
    r"(?:(?i:answer|antwort)|답|答案|答)[\s*:]*(?:(?i:is|ist)|은|는|是|为)?"
    r"|(?i:choice|option|meaning|interpretation|wahl|bedeutung)[\s*]+(?i:is|ist)"
    r"|(?:의미|선택지)[\s*]*(?:은|는)"
    r"|(?:选项|含义|意思)[\s*]*(?:是|为)"
    r")[\s*:]*" + LETTER
)

# "(B)" or "Option B" / "choice B"; the earliest one in the response is taken as the choice.
CHOICE_MARKER = re.compile(r"\(([A-E])\)|(?i:option|choice)\s+([A-E])(?![A-Za-z])")

# A response that is only a letter: "A", "A.", "B)", "C: ..."
BARE_LETTER = re.compile(r"^\s*([A-E])(?:[.):]|\s*$)")

OPTION_LINE = re.compile(r"^\(([A-E])\)\s*(.+)$", re.MULTILINE)


def normalize(text):
    for circled, letter in CIRCLED_NUMBERS.items():
        text = text.replace(circled, letter)
    # NFKC turns full-width characters such as "（Ｂ）" and "：" into "(B)" and ":".
    return unicodedata.normalize("NFKC", text)


def parse_options(prompt):
    """Map each option letter in a test item to its text."""
    return {m.group(1): m.group(2).strip() for m in OPTION_LINE.finditer(normalize(prompt))}


def extract_answer(response, prompt=None):
    """
    Return the option letter (A-E) the model chose, or None if it cannot be determined.

    Checked in order:
      1. An explicit answer label ("Answer: C", "정답은 (C)", "答：B").
      2. The first option marker ("(B)" or "Option B"). Explanations often discuss the
         other options afterwards, so only the first one counts.
      3. A response consisting of just a letter ("A", "B.").
      4. If the prompt is given, a response that is exactly one option's text.
    """
    if not isinstance(response, str) or not response.strip():
        return None

    text = normalize(response)

    match = ANSWER_LABEL.search(text)
    if match:
        return match.group(1)

    match = CHOICE_MARKER.search(text)
    if match:
        return match.group(1) or match.group(2)

    match = BARE_LETTER.match(text)
    if match:
        return match.group(1)

    if prompt is not None:
        cleaned = text.strip().rstrip(".。!").lower()
        for letter, option_text in parse_options(prompt).items():
            if cleaned == option_text.rstrip(".。!").lower():
                return letter

    return None


def pilot_sample(df):
    """10 rows: two per category, with each gold letter appearing exactly twice."""
    letters = "ABCDE"
    picks = []
    for i, category in enumerate(df["type"].unique()):
        for letter in (letters[i], letters[(i + 2) % 5]):
            matching = df[(df["type"] == category) & (df["answer"] == letter)]
            picks.append(matching.sample(1, random_state=0))
    return pd.concat(picks)


def subset_sample(df):
    """90 rows: 18 per category and 18 per gold letter (3 or 4 in each category/letter pair)."""
    letters = "ABCDE"
    picks = []
    for i, category in enumerate(df["type"].unique()):
        for j, letter in enumerate(letters):
            count = 4 if (j - i) % 5 < 3 else 3
            matching = df[(df["type"] == category) & (df["answer"] == letter)]
            picks.append(matching.sample(count, random_state=0))
    return pd.concat(picks).sort_values("id")


SAMPLES = {"pilot": pilot_sample, "subset": subset_sample, "all": lambda df: df}


def run_evaluation(df, languages, query_model, model_name, output_path, trials=1):
    """Query the model and append each result to output_path; rows already there are skipped."""
    done = set()
    if os.path.exists(output_path):
        previous = pd.read_csv(output_path)
        done = set(zip(previous["trial"], previous["language"], previous["id"]))
        print(f"Resuming: {len(done)} results already in {output_path}")

    total = trials * len(languages) * len(df)
    count = 0

    for trial in range(1, trials + 1):
        for language in languages:
            for _, row in df.iterrows():
                count += 1
                if (trial, language, row["id"]) in done:
                    continue

                prompt = row[language]
                start = time.perf_counter()
                reply = query_model(prompt)
                seconds = time.perf_counter() - start
                # Models may return (text, extra columns), e.g. token counts.
                raw_response, extra = reply if isinstance(reply, tuple) else (reply, {})
                prediction = extract_answer(raw_response, prompt)
                print(f"[{count}/{total}] trial {trial} {language} id {row['id']}: "
                      f"predicted {prediction}, gold {row['answer']}")

                result = {
                    "model": model_name,
                    "trial": trial,
                    "id": row["id"],
                    "type": row["type"],
                    "language": language,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "prediction": prediction,
                    "gold": row["answer"],
                    "correct": prediction == row["answer"],
                    "seconds": round(seconds, 2),
                    **extra,
                }
                pd.DataFrame([result]).to_csv(
                    output_path, mode="a", header=not os.path.exists(output_path), index=False
                )

    return pd.read_csv(output_path)


def summarize(results_df):
    """Accuracy (%) per language, laid out like the paper's Table 5."""
    by_type = results_df.pivot_table(index="language", columns="type", values="correct", aggfunc="mean") * 100
    table = by_type.reindex(columns=PRAGMATIC_TYPES)
    # The paper's "Avg." is the mean of the four maxims; literal items are reported separately.
    table["4-maxim avg"] = table[PRAGMATIC_TYPES].mean(axis=1)
    table["literal"] = by_type.get("literal")
    table["no-answer (E)"] = (
        results_df[results_df["gold"] == "E"].groupby("language")["correct"].mean() * 100
    )
    order = [lang for lang in LANGUAGES if lang in table.index]
    return table.loc[order].round(1)


if __name__ == "__main__":
    import argparse

    from models import MODELS

    parser = argparse.ArgumentParser(description="Run MultiPragEval on one model.")
    parser.add_argument("--model", choices=MODELS, default="fake")
    parser.add_argument("--sample", choices=SAMPLES, default="pilot",
                        help="pilot = 10 questions, subset = 90 questions, all = 300 questions; "
                             "pilot and subset are balanced across categories and gold letters")
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=["english"])
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()

    model_name, query_model = MODELS[args.model]
    df = pd.read_csv("test_suite.csv")
    questions = SAMPLES[args.sample](df)

    output_path = f"results/{model_name.replace(':', '_')}_{args.sample}.csv"
    results_df = run_evaluation(questions, args.languages, query_model, model_name,
                                output_path, args.trials)

    print(f"\n{len(results_df)} rows in {output_path}")
    print("Unparsed responses:", results_df["prediction"].isna().sum())
    print("\nAccuracy (%):")
    print(summarize(results_df).to_string())
