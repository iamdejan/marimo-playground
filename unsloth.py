import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # IELTS Task 2 Writing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Dataset
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    df_train_raw = load_dataset("chillies/IELTS-writing-task-2-evaluation", split="train")
    df_train_raw = df_train_raw.to_pandas()
    return df_train_raw, load_dataset


@app.cell
def _(df_train_raw):
    df_train_raw[:50]
    return


@app.cell
def _(load_dataset):
    df_test_raw = load_dataset("chillies/IELTS-writing-task-2-evaluation", split="test")
    df_test_raw = df_test_raw.to_pandas()
    return (df_test_raw,)


@app.cell
def _(df_test_raw):
    df_test_raw[:50]
    return


@app.cell
def _(df_train_raw):
    print(df_train_raw["band"].astype(str).value_counts().head(40).to_string())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Light Text Cleaning
    """)
    return


@app.cell
def _():
    import unicodedata
    import re


    def light_clean(text: str) -> str:
        """
        Light cleaning only:
        - normalize unicode
        - remove odd control chars
        - fix spacing around punctuation
        - preserve learner grammar/spelling mistakes
        """
        if not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFKC", text)

        # Remove weird control chars, keep newline/tab
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

        # Remove spaces before punctuation (including period)
        text = re.sub(r"\s+([,;:!?.])", r"\1", text)

        # Add a missing space after punctuation ONLY when next char is a letter
        # Avoid breaking decimals like 7.5 or thousands like 1,000
        text = re.sub(r"([,;:!?])([A-Za-z])", r"\1 \2", text)

        # Word (2+ letters) glued to next sentence: "Mr.Smith" / "word.Next" -> add space
        text = re.sub(r"(?<=[A-Za-z][A-Za-z])\.(?=[A-Za-z])", ". ", text)

        # Abbreviation period followed by a real word (2+ lowercase): "e.g.this" -> "e.g. this"
        # Won't break abbreviation internals like "e.g." or "U.S."
        text = re.sub(r"(?<=[A-Za-z])\.(?=[a-z]{2,})", ". ", text)

        # Remove extra space after opening brackets
        text = re.sub(r"([\(\[\{])\s+", r"\1", text)

        # Remove extra space before closing brackets
        text = re.sub(r"\s+([\)\]\}])", r"\1", text)

        # Collapse repeated spaces/tabs
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


    # Quick verification for risky cases
    test_cases = [
        "Nowadays  ,many people believe that education is important .",
        "Band 7.5",
        "The value is 3.14 and the fee is 1,000 dollars.",
        "e.g.this should stay sensible.",
        "Mr.Smith went home",
        "( word )  and [ another ]",
        "U.S. is a country",
        "U.K. and U.S.A. are countries",
        "word.Next sentence here",
        "end.Start of new sentence",
        "U.S.government should work",
        "i.e.something like this",
    ]
    print("=== CLEANING VERIFICATION ===")
    for t in test_cases:
        print("BEFORE:", t)
        print("AFTER :", light_clean(t))
        print()
    return light_clean, re


@app.cell
def _(df_test_raw, df_train_raw, light_clean):
    df_train = df_train_raw.copy()
    df_test = df_test_raw.copy()

    for col in ["prompt", "essay", "evaluation"]:
        df_train[f"{col}_clean"] = df_train[col].apply(light_clean)
        df_test[f"{col}_clean"] = df_test[col].apply(light_clean)
    return df_test, df_train


@app.cell
def _(df_train):
    df_train[:30]
    return


@app.cell
def _(df_test):
    df_test[:30]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Band Normalization
    """)
    return


@app.cell
def _(re):
    import numpy as np
    import pandas as pd


    def normalize_band(val) -> float:
        """
        Normalize band labels to float.
        Examples:
        - "7", "7.0", "Band 7", "band score of 7.5" -> 7.0 / 7.5
        - "<4", "below 4" -> 3.5
        - ">6", "above 6" -> 6.5
        Returns np.nan if not parseable.
        """
        if pd.isna(val):
            return np.nan

        s = str(val).strip().lower()

        # Below-threshold
        if re.match(r"^<\s*4$", s) or s in {"<4", "below 4", "less than 4", "below band 4"}:
            return 3.5

        # Above-threshold
        m = re.match(r"^(?:>|above|more than)\s*(\d+(?:\.\d+)?)$", s)
        if m:
            return float(m.group(1)) + 0.5

        # Bracketed value
        m = re.search(r"\[(\d+(?:\.\d+)?)\]", s)
        if m:
            v = float(m.group(1))
            return v if 0 <= v <= 9 else np.nan

        # "Band 7", "band score of 7.5"
        m = re.search(r"band\s*(?:score\s*)?(?:of\s*)?(\d+(?:\.\d+)?)", s)
        if m:
            v = float(m.group(1))
            return v if 0 <= v <= 9 else np.nan

        # Exact numeric
        m = re.match(r"^(\d+(?:\.\d+)?)$", s)
        if m:
            v = float(m.group(1))
            return v if 0 <= v <= 9 else np.nan

        # Last-resort numeric search
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if m:
            v = float(m.group(1))
            return v if 0 <= v <= 9 else np.nan

        return np.nan


    test_bands = [
        "7",
        "7.0",
        "7.5",
        "Band 7",
        "band score of 7.5",
        "Band Score: 6",
        "<4",
        "below 4",
        "8.5",
        "5.0",
        ">6",
        "garbage",
    ]
    print("=== BAND NORMALIZATION VERIFICATION ===")
    for b in test_bands:
        print(f"{str(b):25s} -> {normalize_band(b)}")
    return (normalize_band,)


@app.cell
def _(df_test, df_train, normalize_band):
    df_train["band_numeric"] = df_train["band"].apply(normalize_band)
    df_test["band_numeric"] = df_test["band"].apply(normalize_band)
    return


@app.cell
def _(df_train):
    df_train[:30]
    return


@app.cell
def _(df_test):
    df_test[:30]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
