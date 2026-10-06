import sys

import pandas as pd

from evaluate import extract_answer

df = pd.read_csv("test_suite.csv").set_index("id")
english_prompt = df.loc[2, "english"]
german_prompt = df.loc[2, "german"]

# (description, response, expected letter, prompt passed to the parser)
CASES = [
    # Responses quoted in the paper's Appendix D.
    ("paper: GPT-4, id 7", (
        'The most appropriate meaning of Emily\'s statement "Either he comes or he doesn\'t" is:\n'
        "(B) Charlie may or may not come.\n"
        "This interpretation is based on the literal meaning of the statement, which indicates "
        "uncertainty or neutrality regarding Charlie's attendance."
    ), "B", None),
    ("paper: Claude3-Opus, id 7 (mentions every other letter)", (
        'The most appropriate meaning of Emily\'s utterance "Either he comes or he doesn\'t" is:\n'
        "(A) Emily doesn't care if Charlie comes or not.\n"
        "Explanation:\n"
        "Emily's statement suggests indifference towards Charlie's attendance at the club meeting.\n"
        "The other options are less fitting because:\n"
        "(B) While the statement does suggest uncertainty about Charlie's attendance, it doesn't "
        "capture Emily's indifference.\n"
        "(C) There is no indication in the given context that Emily has decided to apologize to Charlie.\n"
        "(D) The statement does not reflect on Emily's speaking abilities.\n"
        "(E) Option (A) is the most appropriate interpretation of the given utterance."
    ), "A", None),
    ("paper: Llama2-13B, id 127", (
        "Answer:\n(C) Charlie plans to drink.\nExplanation:\n"
        'The statement "I drove here" is often used as a way of saying that one plans to drink alcohol.'
    ), "C", None),
    ("paper: Llama2-7B, id 127 (reasoning contradicts choice)", (
        "Answer: (C) Charlie plans to drink.\nExplanation:\n"
        "It implies that the speaker arrived at the location by driving, and therefore, "
        "they are not planning to drink. Therefore, option (C) is the most appropriate meaning."
    ), "C", None),
    ("paper: Llama2-13B, id 75, Korean circled digit", (
        '답: ① 철수는 바쁘게 부지런히 일한다.\n'
        '설명: 위 발화 "쟤는 숨도 안 쉬어"는 "철수는 바쁘게 부지런히 일한다"라는 의미를 갖고 있습니다.'
    ), "A", None),
    ("paper: Llama2-13B, id 75, Chinese", "答：B", "B", None),
    ("paper: GPT-4, id 169", (
        'The most appropriate meaning of Charlie\'s utterance "I have a gift card" in the context provided is:\n'
        "(E) None of the above.\n"
        "The correct interpretation, which isn't listed among the options, is likely that Charlie "
        "is indicating he has a gift card to use at the cafe."
    ), "E", None),
    ("paper: Claude3-Opus, id 169", (
        'The most appropriate meaning of Charlie\'s utterance "I have a gift card" is:\n'
        "(E) None of the above.\n"
        "The other options (A, B, C, and D) do not accurately represent the meaning of Charlie's "
        "statement in the given context."
    ), "E", None),

    # Simple formats from the original plan.
    ("plain letter", "A", "A", None),
    ("letter in parentheses", "(A)", "A", None),
    ("Answer: A", "Answer: A", "A", None),
    ("The answer is A.", "The answer is A.", "A", None),
    ("Option B ...", "Option B is the most appropriate response.", "B", None),
    ("letter with period", "D.", "D", None),

    # Responses whose first word starts with a capital A-E (the original parser misread these).
    ("starts with 'Based'", "Based on the context, the professor means (A).", "A", None),
    ("starts with 'Emily'", "Emily is being sarcastic, so the answer is (A).", "A", None),
    ("starts with 'Definitely'", "Definitely (C).", "C", None),
    ("starts with 'Considering'", "Considering the context, (D) fits best.", "D", None),
    ("starts with article 'A'", "A tautology like this signals a refusal, so (A) is correct.", "A", None),

    # Markdown formatting, as produced by Gemini 3.x.
    ("Gemini: bold option after label", "The correct answer is **(A) Emily doesn't care if Charlie comes or not.**", "A", None),
    ("Gemini: bold option, no label", "The most appropriate meaning is **(B) Charlie may or may not come.**", "B", None),
    ("bold label", "**Answer:** (C)", "C", None),
    ("walks through options, then labels the answer", (
        "To determine the most appropriate meaning, consider each option.\n"
        "* (A) is too literal.\n* (B) ignores the context.\n* (C) fits the situation.\n\n"
        "The correct answer is **(C)**."
    ), "C", None),

    # Reasoning first, conclusion last (real Gemini 3.8 Flash replies, shortened).
    ("Gemini english id 240: 'correct choice is'", (
        "Looking at the options:\n- (A) Incorrect (the boss is critical of the report).\n"
        "- (B) Incorrect (no mention of a business trip).\n- (E) Correct.\n\n"
        "Therefore, the correct choice is **(E) None of the above.**"
    ), "E", None),
    ("Gemini english id 283: 'choice is:' then a hedge", (
        "- **(A)**, **(B)**, and **(C)** introduce assumptions that have no basis in the text.\n"
        "- **(D)** adds the detail that the romance novel was \"new\".\n\n"
        "Because option (D) introduces an unverified detail, the most strictly accurate choice is:\n\n"
        "**(E) None of the above.**\n\n*(Note: ... **(D)** is the only option that directly relates ...)*"
    ), "E", None),
    ("Gemini chinese id 233: '选项是' after '各选项：'", (
        "现在分析各选项：\n* (A) 挥手的人是张伟的叔叔 —— 错误，那是他的父亲。\n"
        "* (E) 没有正确答案 —— 正确。\n\n因此，最恰当的选项是 **(E)**。"
    ), "E", None),
    ("Gemini chinese id 293: '选项是' and '正确答案：' agree", (
        "张伟**没有**看成这部电影（排除了选项 A）。\n选项 (C) 与他的态度完全相反，明显错误。\n\n"
        "因此，最恰当地表达张伟话语潜台词/含义的选项是 **(D)**。\n\n**正确答案：(D) 张伟委婉地表示不想和王芳一起看新片。**"
    ), "D", None),
    ("'Option (A) is incorrect' is not an answer", (
        "Option (A) is incorrect because it is literal. Option (B) ignores the context. "
        "Therefore, the most appropriate meaning is (C)."
    ), "C", None),
    ("German 'Bedeutung ist'", "(A) ist falsch. (B) ist zu wörtlich. Die passendste Bedeutung ist **(C)**.", "C", None),
    ("Korean '의미는'", "(A)는 틀렸습니다. 따라서 가장 적절한 의미는 **(D)** 입니다.", "D", None),

    # Other languages and full-width characters.
    ("German label", "Die richtige Antwort ist (A).", "A", None),
    ("German short", "Antwort: B", "B", None),
    ("Korean label", "정답은 (C)입니다.", "C", None),
    ("Chinese label", "答案是D。", "D", None),
    ("Chinese full-width", "答案：（Ｅ）", "E", None),

    # No letter at all: fall back to matching an option's exact text.
    ("option text only (English)", "None of the above.", "E", english_prompt),
    ("option text only (German)", "Keine der obigen Aussagen ist richtig.", "E", german_prompt),
    ("option text only, no period", "The deadline can't be extended because rules must be followed", "A", english_prompt),

    # Cannot be determined: must return None, not a guess.
    ("no answer given", "I'm not sure what the speaker means here.", None, None),
    ("'answer is' followed by a word", "The answer is Emily's frustration with the situation.", None, None),
    ("quotes the utterance only", "Rules are rules means the professor is strict.", None, english_prompt),
    ("empty response", "", None, None),
    ("missing response", None, None, None),
]

failures = 0
for description, response, expected, prompt in CASES:
    got = extract_answer(response, prompt)
    ok = got == expected
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {description:55s} expected={expected!s:5s} got={got}")

print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
sys.exit(1 if failures else 0)
