import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Data
    """)
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv("data/salary.csv")
    df
    return (df,)


@app.cell
def _(df):
    # Convert salary to integers
    df["Salary"] = df["Salary"].replace(",", "", regex=True)
    df["Salary"] = df["Salary"].astype("int")
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Incomplete data/Missing values
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df["Compensation"].isna().sum()
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1 Dropping columns with too many missing values
    """)
    return


@app.cell
def _(df):
    df.drop(columns=["Context", "Other currency", "Salary context"], inplace=True)
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2 Dropping missing values

    - We can drop missing values here using dropna() function.
    - If we do dropna() without specifying any parameter, it will drop rows that has even 1 missing values
    - If we do dropna(thresh=2) it will drop rows that has more than 2 missing values.
    - If we do dropna(how=all) it will drop only rows where all values are missing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare the number of rows before and after dropna()
    """)
    return


@app.cell
def _(df):
    len(df)
    return


@app.cell
def _(df):
    df.dropna(inplace=True)
    return


@app.cell
def _(df):
    len(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 Impute missing values

    - A simple method where we don't analyse per column.
    - We can impute missing values for all related columns using most frequent.
    - Imputed values can be the mean, median. This is changed in the 'strategy' parameter.
    - There are other simple methods for imputing missing values. Eg ll.na()
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use `sklearn.impute.SimpleImputer` to complete missing values with simple strategies.
    """)
    return


@app.cell
def _(df):
    from sklearn.impute import SimpleImputer

    most_frequent_imputer = SimpleImputer(strategy="most_frequent")
    df.iloc[:, :] = most_frequent_imputer.fit_transform(df)
    df.head()
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Noisy data

    - Noise can be random errors and/or outliers.
    - It can also be non-sensible data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Remove outliers

    - Use boxplot or histograms to nd outliers (`df.boxplot()` or `df.hist()`)
    - Outliers are values that are waaay higher or lower than most of the data. Usually there are not many data points like this. So we can consider this as noise
    """)
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.boxplot(column=["Salary", "Compensation"])
    return


@app.cell
def _(df):
    df.plot.hist(column=["Salary"])
    return


@app.cell
def _(df, pd):
    from scipy import stats
    import numpy as np

    zscore = stats.zscore(df["Salary"])
    zscore_df = pd.DataFrame(zscore)
    zscore_df.describe()
    return np, stats


@app.cell
def _(df, np, stats):
    df.drop(df[np.abs(stats.zscore(df["Salary"])) >= 3].index, inplace=True)
    df.drop(df[np.abs(stats.zscore(df["Compensation"])) >= 3].index, inplace=True)
    df.boxplot(column=["Salary", "Compensation"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Remove non-sensible values
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df[df["Compensation"] > 56000]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Inconsistent data

    - Find 'duplicated' rows. For example 'Academia' , 'Academic', 'Academia/Research' should be the same. One way is to use fuzzy matching (fuzzywuzzy package)
    - Find typos
    - Warning , this is very tedious
    - [Ref] https://www.kaggle.com/code/ramjan135/data-cleaning-challenge-inconsistent-data-entry
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.1 Check for each columns. Example is 'Gender'

    - Print out the unique items for each columns using df['column].unique()
    - You'll see some are not that unique. Using the 'Gender' column, there is 'Other or prefer not to answer' and 'Prefer not to answer'. Let's just group them into 'Other'
    - Try checking columns with the lowest number of unique items first
    """)
    return


@app.cell
def _(df, pd):
    pd.DataFrame(df["Gender"].unique(), columns=["Gender"])
    return


@app.cell
def _(df, pd):
    df["Gender"] = df["Gender"].replace("Prefer not to answer", "Other")
    df["Gender"] = df["Gender"].replace("Other or prefer not to answer", "Other")
    pd.DataFrame(df["Gender"].unique(), columns=["Gender"])
    return


if __name__ == "__main__":
    app.run()
