# charts.py
import altair as alt
import pandas as pd
import os
# Load your dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "food-waste-per-capita.csv")

df = pd.read_csv(file_path)
# Ensure numeric types
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
for col in ["Retail", "home consumption", "Households"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Constants
ENT_COL = "Entity"
YEAR_COL = "Year"
VALUE_COL = "Retail"  # Default metric


# -------------------------------
# 1) Line chart with highlight
# -------------------------------
def line_trend_highlight():
    opts = sorted(df[ENT_COL].dropna().unique().tolist())
    sel = alt.selection_single(
        fields=[ENT_COL],
        bind=alt.binding_select(options=opts, name=f"{ENT_COL} "),
        empty="all"
    )

    base = (
        alt.Chart(df)
        .mark_line(opacity=0.5)
        .encode(
            x=alt.X(f"{YEAR_COL}:Q"),
            y=alt.Y(f"{VALUE_COL}:Q", title=VALUE_COL),
            color=alt.value("#cfcfcf"),
            detail=ENT_COL
        )
    )

    hi = (
        alt.Chart(df)
        .mark_line(size=3)
        .encode(
            x=f"{YEAR_COL}:Q",
            y=f"{VALUE_COL}:Q",
            color=alt.Color(f"{ENT_COL}:N", legend=None),
            tooltip=[ENT_COL, YEAR_COL, VALUE_COL]
        )
        .transform_filter(sel)
    )

    return (base + hi).add_params(sel).properties(
        width=700,
        height=360,
        title=f"{VALUE_COL} trend (highlight selection)"
    )


# -------------------------------
# 2) Top N bar chart by year
# -------------------------------
def topn_bar_by_year(top_n=15):
    year_opts = sorted(df[YEAR_COL].dropna().unique().astype(int).tolist())
    sel_year = alt.selection_single(
        fields=[YEAR_COL],
        bind=alt.binding_select(options=year_opts, name="Year "),
        empty="none"
    )

    filtered = alt.Chart(df).transform_filter(sel_year)

    ranked = filtered.transform_window(
        rank="rank()",
        sort=[alt.SortField(VALUE_COL, order="descending")],
        groupby=[]
    ).transform_filter(alt.datum.rank <= top_n)

    return (
        ranked.mark_bar()
        .encode(
            y=alt.Y(f"{ENT_COL}:N", sort="-x", title=""),
            x=alt.X(f"{VALUE_COL}:Q", title=VALUE_COL),
            tooltip=[ENT_COL, alt.Tooltip(f"{VALUE_COL}:Q", format=",.2f")]
        )
        .add_params(sel_year)
        .properties(width=700, height=25 * top_n, title=f"Top {top_n} by {VALUE_COL}")
    )


# -------------------------------
# 3) Simple line chart
# -------------------------------
def simple_line(metric="Retail"):
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=f"{YEAR_COL}:Q",
            y=f"{metric}:Q",
            color=ENT_COL
        )
        .properties(width=700, height=360, title=f"{metric} — Simple Trend")
    )


# -------------------------------
# 4) Grouped bar chart
# -------------------------------
def grouped_bar(metric="Households", year=2020):
    subset = df[df[YEAR_COL] == year]
    return (
        alt.Chart(subset)
        .mark_bar()
        .encode(
            x=alt.X(f"{ENT_COL}:N", sort="-y"),
            y=f"{metric}:Q",
            color=ENT_COL
        )
        .properties(width=700, height=360, title=f"{metric} — {year} Grouped Bar")
    )


# -------------------------------
# 5) Scatter plot
# -------------------------------
def scatter_plot(x_metric="Retail", y_metric="home consumption", year=2020):
    subset = df[df[YEAR_COL] == year]
    return (
        alt.Chart(subset)
        .mark_circle(size=100)
        .encode(
            x=f"{x_metric}:Q",
            y=f"{y_metric}:Q",
            color=ENT_COL,
            tooltip=[ENT_COL, x_metric, y_metric]
        )
        .properties(width=700, height=360, title=f"{x_metric} vs {y_metric} — {year}")
    )


# -------------------------------
# 6) Area chart
# -------------------------------
def area_chart(metric="Retail"):
    return (
        alt.Chart(df)
        .mark_area(opacity=0.5)
        .encode(
            x=f"{YEAR_COL}:Q",
            y=f"{metric}:Q",
            color=ENT_COL
        )
        .properties(width=700, height=360, title=f"{metric} — Area Chart")
    )


# -------------------------------
# 7) Histogram
# -------------------------------
def histogram(metric="Retail", year=2020):
    subset = df[df[YEAR_COL] == year]
    return (
        alt.Chart(subset)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=20)),
            y='count()',
            tooltip=['count()']
        )
        .properties(width=700, height=360, title=f"Distribution of {metric} — {year}")
    )



