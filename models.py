import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

# The paper used temperature 0.5 for every model.
TEMPERATURE = 0.5

GEMINI_MODEL = "gemini-3.8-flash"
LLAMA_MODEL = "llama3:8b-instruct-q4_K_M"
# Prompt plus reply stay well under 1,000 tokens; a smaller context window keeps memory use down on 16 GB.
LLAMA_CONTEXT_TOKENS = 2048

MAX_RETRIES = 8

_gemini_client = None


def query_fake(prompt):
    """Always answers "A"; used to check the pipeline (expected accuracy: exactly 20%)."""
    return "A"


def _query_gemini(prompt, thinking):
    """Return the reply text and token counts. thinking=True keeps the model's default thinking."""
    from google import genai
    from google.genai import types

    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        thinking_config=None if thinking else types.ThinkingConfig(thinking_budget=0),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config
            )
            usage = response.usage_metadata
            return response.text, {
                "input_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
                "thinking_tokens": usage.thoughts_token_count or 0,
            }
        except genai.errors.APIError as error:
            # Only rate limits and server errors are worth retrying.
            if error.code not in (429, 500, 503) or attempt == MAX_RETRIES:
                raise
            suggested = re.search(r"retry in ([\d.]+)s", str(error))
            wait = float(suggested.group(1)) + 1 if suggested else 2 ** attempt
            print(f"  Gemini error {error.code} {error.status}; retrying in {wait:.0f}s")
            time.sleep(wait)


def query_gemini(prompt):
    # The paper's 2024 models did not reason before answering, so thinking is disabled.
    return _query_gemini(prompt, thinking=False)


def query_gemini_thinking(prompt):
    return _query_gemini(prompt, thinking=True)


def query_llama(prompt):
    import ollama

    response = ollama.chat(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": TEMPERATURE, "num_ctx": LLAMA_CONTEXT_TOKENS},
    )
    return response["message"]["content"]


MODELS = {
    "fake": ("fake-always-A", query_fake),
    "gemini": (GEMINI_MODEL, query_gemini),
    "gemini-thinking": (f"{GEMINI_MODEL}-thinking", query_gemini_thinking),
    "llama": (LLAMA_MODEL, query_llama),
}
