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
    # Data Preprocessing Part 2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load dataframes
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    df_salary = pd.read_csv("data/salary_cleaned.csv")
    df_salary.drop_duplicates(inplace=True)
    df_salary
    return df_salary, pd


@app.cell
def _(df_salary):
    df_salary["Salary"].isna().sum()
    return


@app.cell
def _(df_salary):
    df_salary.dropna(inplace=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Integration
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load New dataframe
    """)
    return


@app.cell
def _(pd):
    df_xtra_info = pd.read_csv("data/xtra_info.csv")
    df_xtra_info.drop_duplicates(inplace=True)
    df_xtra_info
    return (df_xtra_info,)


@app.cell
def _(df_xtra_info):
    df_xtra_info.dropna(inplace=True)
    return


@app.cell
def _(df_xtra_info):
    df_xtra_info.info()
    return


@app.cell
def _(df_salary, df_xtra_info, pd):
    merged_df = pd.merge(df_salary, df_xtra_info, how="inner", left_on="USERID", right_on="USERID")
    merged_df.drop_duplicates(inplace=True)
    merged_df
    return (merged_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2 Drop Derivable data

    Identify which columns is derivable and drop that column.
    """)
    return


@app.cell(hide_code=True)
def _(merged_df):
    merged_df.info()
    return


@app.cell
def _(merged_df):
    merged_df.drop(columns="Salary_month_USD", inplace=True)
    merged_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 Object Identification

    Convert Special Bonus_GBP column to USD and rename 'Special Bonus_GBP' to 'Special_Bonus_USD'.
    """)
    return


@app.cell
def _(merged_df):
    # as per 2026-04-19, GBP 1 = USD 1.35
    gbp_to_usd = 1.35

    merged_df["Special_Bonus_USD"] = 1.35 * merged_df["Special Bonus_GBP"]
    merged_df.drop(columns="Special Bonus_GBP", inplace=True)
    merged_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Data transformation - Normalization

    - Min-max normalization
    - Zscore/Standard normalization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Min-max normalization
    """)
    return


@app.cell
def _(merged_df):
    from sklearn.preprocessing import MinMaxScaler

    salary_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_salary = salary_scaler.fit_transform(merged_df[["Salary"]])
    scaled_salary
    return


@app.cell
def _(merged_df):
    from sklearn.preprocessing import StandardScaler

    standard_scaler = StandardScaler()
    scaled_bonus = standard_scaler.fit_transform(merged_df[["Special_Bonus_USD"]])
    scaled_bonus
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Data Reduction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.1 Aggregation

    Let's say we want to groupby city. Different cities should have different salary ranges. New York vs Wisconsin etc.
    """)
    return


@app.cell
def _(merged_df):
    grouped_salary_by_city = merged_df.groupby(by="City")
    salary_range = grouped_salary_by_city["Salary"].agg(["min", "max"])
    salary_range
    return (salary_range,)


@app.cell
def _(salary_range):
    salary_range.loc[["New York", "Wisconsin"]]
    return


@app.cell
def _(salary_range):
    salary_range.sort_values(by="min", ascending=False)
    return


if __name__ == "__main__":
    app.run()
