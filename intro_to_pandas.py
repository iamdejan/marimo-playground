import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Intro to pandas
    """)
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic Concepts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The primary data structures in *pandas* are implemented as two classes:

    - **`DataFrame`**, which you can imagine as a relational data table, with rows and named columns.
    - **`Series`**, which is a single column. A `DataFrame` contains one or more `Series` and a name for each `Series`.

    The data frame is a commonly used abstraction for data manipulation. Similar implementations exist in [Spark](https://spark.apache.org/) and [R](https://www.r-project.org/about.html).

    One way to create a `Series` is to construct a `Series` object. For example:
    """)
    return


@app.cell
def _(pd):
    city_names = pd.Series(["San Francisco", "San Jose", "Sacramento"])
    population = pd.Series([852469, 1015785, 485199])

    pd.DataFrame({"City name": city_names, "Population": population})
    return city_names, population


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    But most of the time, you load an entire file into a `DataFrame`. The following example loads a file with California housing data. Run the following cell to load the data and create feature definitions:
    """)
    return


@app.cell
def _(pd):
    california_housing_dataframe = pd.read_csv(
        "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=","
    )
    california_housing_dataframe.describe()
    return (california_housing_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The example above used `DataFrame.describe` to show interesting statistics about a `DataFrame`. Another useful function is `DataFrame.head`, which displays the first few records of a `DataFrame`:
    """)
    return


@app.cell
def _(california_housing_dataframe):
    california_housing_dataframe.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another powerful feature of *pandas* is graphing. For example, `DataFrame.hist` lets you quickly study the distribution of values in a column:
    """)
    return


@app.cell
def _(california_housing_dataframe):
    import matplotlib.pyplot as plt

    california_housing_dataframe.hist("housing_median_age")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accessing Data

    You can access `DataFrame` data using familiar Python dict/list operations:
    """)
    return


@app.cell
def _(city_names, pd, population):
    cities = pd.DataFrame({"City name": city_names, "Population": population})
    print(type(cities["City name"]))
    cities["City name"]
    return (cities,)


@app.cell
def _(cities):
    print(type(cities["City name"][1]))
    cities["City name"][1]
    return


@app.cell
def _(cities):
    print(type(cities[0:2]))
    cities[0:2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In addition, *pandas* provides an extremely rich API for advanced [indexing and selection](http://pandas.pydata.org/pandas-docs/stable/indexing.html) that is too extensive to be covered here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manipulating Data

    You may apply Python's basic arithmetic operations to `Series`. For example:
    """)
    return


@app.cell
def _(population):
    population / 1000.0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [NumPy](http://www.numpy.org/) is a popular toolkit for scientific computing. *pandas* `Series` can be used as arguments to most NumPy functions:
    """)
    return


@app.cell
def _(population):
    import numpy as np

    np.log(population)
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For more complex single-column transformations, you can use `Series.apply`. Like the Python [map function](https://docs.python.org/2/library/functions.html#map),
    `Series.apply` accepts as an argument a [lambda function](https://docs.python.org/2/tutorial/controlflow.html#lambda-expressions), which is applied to each value.

    The example below creates a new `Series` that indicates whether `population` is over one million:
    """)
    return


@app.cell
def _(population):
    population.apply(lambda val: val > 1000000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Modifying `DataFrames` is also straightforward. For example, the following code adds two `Series` to an existing `DataFrame`:
    """)
    return


@app.cell
def _(cities, pd):
    cities["Area square miles"] = pd.Series([46.87, 176.53, 97.92])
    cities["Population density"] = cities["Population"] / cities["Area square miles"]
    cities
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise #1

    Modify the `cities` table by adding a new boolean column that is True if and only if *both* of the following are True:

      * The city is named after a saint.
      * The city has an area greater than 50 square miles.

    **Note:** Boolean `Series` are combined using the bitwise, rather than the traditional boolean, operators. For example, when performing *logical and*, use `&` instead of `and`.

    **Hint:** "San" in Spanish means "saint."
    """)
    return


@app.cell
def _(cities):
    cities["New column"] = (cities["City name"].apply(lambda c: c.startswith("San"))) & (
        cities["Area square miles"] > 50.0
    )
    cities["New column"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Indexes
    Both `Series` and `DataFrame` objects also define an `index` property that assigns an identifier value to each `Series` item or `DataFrame` row.

    By default, at construction, *pandas* assigns index values that reflect the ordering of the source data. Once created, the index values are stable; that is, they do not change when data is reordered.
    """)
    return


@app.cell
def _(city_names):
    city_names.index
    return


@app.cell
def _(cities):
    cities.index
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Call `DataFrame.reindex` to manually reorder the rows. For example, the following has the same effect as sorting by city name:
    """)
    return


@app.cell
def _(cities):
    cities.reindex([2, 0, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reindexing is a great way to shuffle (randomize) a `DataFrame`. In the example below, we take the index, which is array-like, and pass it to NumPy's `random.permutation` function, which shuffles its values in place. Calling `reindex` with this shuffled array causes the `DataFrame` rows to be shuffled in the same way.
    Try running the following cell multiple times!
    """)
    return


@app.cell
def _(cities, np):
    cities.reindex(np.random.permutation(cities.index))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise #2

    The `reindex` method allows index values that are not in the original `DataFrame`'s index values. Try it and see what happens if you use such values! Why do you think this is allowed?
    """)
    return


@app.cell
def _(cities):
    cities.reindex([0, 2, 4, 5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This behavior is desirable because indexes are often strings pulled from the actual data (see the [*pandas* reindex
    documentation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reindex.html) for an example
    in which the index values are browser names).

    In this case, allowing "missing" indices makes it easy to reindex using an external list, as you don't have to worry about
    sanitizing the input.
    """)
    return


if __name__ == "__main__":
    app.run()
