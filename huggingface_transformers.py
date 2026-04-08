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
    # A Tour of Transformer Applications
    """)
    return


@app.cell
def _():
    text = """Dear Amazon, last week I ordered an Optimus Prime action figure
    from your online store in Germany. Unfortunately, when I opened the package,
    I discovered to my horror that I had been sent an action figure of Megatron
    instead! As a lifelong enemy of the Decepticons, I hope you can understand my
    dilemma. To resolve the issue, I demand an exchange of Megatron for the
    Optimus Prime figure I ordered. Enclosed are copies of my records concerning
    this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
    return (text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Text Classifier
    """)
    return


@app.cell
def _():
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f",
    )
    return classifier, pipeline


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(classifier, pd, text):
    def classify_text():
        outputs = classifier(text)
        return pd.DataFrame(outputs)


    classify_text()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Named Entity Recognition
    """)
    return


@app.cell
def _(pd, pipeline, text):
    def tag_named_entity_recognition():
        ner_tagger = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            revision="4c53496",
            aggregation_strategy="simple",
        )
        outputs = ner_tagger(text)
        return pd.DataFrame(outputs)


    tag_named_entity_recognition()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question Answering
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In question answering, we provide the model with a passage of text called the _context_,
    along with a question whose answer we’d like to extract. The model then returns the
    span of text corresponding to the answer. Let’s see what we get when we ask a specific
    question about our customer feedback:
    """)
    return


@app.cell
def _():
    model = "Qwen/Qwen2.5-1.5B-Instruct"
    return (model,)


@app.cell
def _(model, pipeline, text):
    def question_answering():
        pipe = pipeline("text-generation", model=model)

        messages = [
            {"role": "system", "content": f"Answer based on this context: {text}"},
            {"role": "user", "content": "What does the customer want?"},
        ]
        result = pipe(messages, max_new_tokens=100)
        print(result[0]["generated_text"][-1]["content"])


    question_answering()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summarization
    """)
    return


@app.cell
def _(model, pipeline, text):
    def summarization():
        pipe = pipeline("text-generation", model=model)

        messages = [
            {"role": "system", "content": f"Answer based on this context: {text}"},
            {"role": "user", "content": "Summarize the text from the customer."},
        ]
        result = pipe(messages, max_new_tokens=100)
        print(result[0]["generated_text"][-1]["content"])


    summarization()
    return


if __name__ == "__main__":
    app.run()
