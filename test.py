import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    x = 7
    x
    return (x,)


@app.cell
def _(x):
    y = x + 1
    y
    return


@app.cell
def _(mo):
    s = mo.ui.slider(start=1, stop=10, step=2)
    s
    return (s,)


@app.cell
def _(mo, s):
    import math

    mo.md(f"$e^{s.value} = {math.exp(s.value):0.4f}$")
    return


if __name__ == "__main__":
    app.run()
