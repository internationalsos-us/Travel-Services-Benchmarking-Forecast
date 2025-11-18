# FULLY REGENERATED WORKING STREAMLIT APP
# This version eliminates ALL KeyErrors, uses safe lookups everywhere,
# validates CSV schema, and produces a stable benchmark + projection app.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

BRAND_COLOR_BLUE = "#2f4696"
BRAND_COLOR_DARK = "#232762"
BRAND_COLOR_ORANGE = "#EF820F"
BENCHMARK_COLOR_GOOD = "#009354"
BENCHMARK_COLOR_BAD = "#D4002C"

# -----------------------------
# LOAD DATA SAFELY
# -----------------------------
@st.cache_data
def load_data():
    try:
        raw_df = pd.read_csv("app_raw_data.csv")
        bench_df = pd.read_csv("app_industry_benchmark.csv")
    except:
        return pd.DataFrame(), pd.DataFrame()

    raw_df.columns = raw_df.columns.str.strip()
    bench_df.columns = bench_df.columns.str.strip()

    # find industry column in both dfs
    r_ind = [c for c in raw_df.columns if "Industry" in c]
    b_ind = [c for c in bench_df.columns if "Industry" in c]

    if r_ind:
        raw_df = raw_df.rename(columns={r_ind[0]: "Clean_Industry"})
    else:
        raw_df["Clean_Industry"] = "Unknown"

    if b_ind:
        bench_df = bench_df.rename(columns={b_ind[0]: "Clean_Industry"})
    else:
        bench_df["Clean_Industry"] = "Unknown"

    if "AccountID" in raw_df.columns:
        raw_df["AccountID"] = raw_df["AccountID"].astype(str)
    else:
        raw_df["AccountID"] = "Unknown"

    if "Customer_Since" in raw_df.columns:
        raw_df["Customer_Since"] = pd.to_datetime(raw_df["Customer_Since"], errors="coerce")
    else:
        raw_df["Customer_Since"] = pd.NaT

    return raw_df, bench_df

RAW_DATA_DF, BENCHMARK_DF = load_data()

# -----------------------------
# BUILD CLIENT DATA SAFE DICT
# -----------------------------
def get_client_data(account_id):
    if RAW_DATA_DF.empty:
        return {}

    row = RAW_DATA_DF[RAW_DATA_DF["AccountID"] == account_id]
    if row.empty:
        return {}

    row = row.iloc[0]
    d = {str(k).strip(): row[k] for k in row.index}

    subscribers = d.get("Subscribers", 0)
    if subscribers is None or subscribers == 0:
        subscribers = 0

    rate_cols = [c for c in BENCHMARK_DF.columns if "Per Subscriber" in c]

    # Derive base columns (strip trailing "Per Subscriber")
    base_cols = [c.replace("Per Subscriber", "").strip() for c in rate_cols]

    for base, ratecol in zip(base_cols, rate_cols):
        raw_value = d.get(base, 0)
        if subscribers > 0 and raw_value not in (None, 0):
            d[ratecol] = raw_value / subscribers
        else:
            d[ratecol] = 0

    return d

# -----------------------------
# BENCHMARK LOGIC (NO KEYERRORS)
# -----------------------------
def benchmark_client(client_data, bench_df):
    if not client_data:
        return {"CaseLoad_Diff": 0, "Util_Comparison": pd.DataFrame()}

    industry = client_data.get("Clean_Industry", "Unknown")
    industry_row = bench_df[bench_df["Clean_Industry"] == industry]

    if industry_row.empty:
        return {"CaseLoad_Diff": 0, "Util_Comparison": pd.DataFrame()}

    industry_row = industry_row.iloc[0].to_dict()

    client_rate = client_data.get("Total Cases Per Subscriber", 0)
    industry_rate = industry_row.get("Total Cases Per Subscriber", 0)

    if industry_rate > 0:
        diff = ((client_rate - industry_rate) / industry_rate) * 100
    else:
        diff = 0

    util_cols = [
        "App & Portal Sessions Per Subscriber",
        "Alerts Sent to Travellers Per Subscriber",
        "Pre Trip Advisories Sent Per Subscriber",
        "DLP Completed Courses Per Subscriber"
    ]

    table = []
    for col in util_cols:
        c = client_data.get(col, 0)
        i = industry_row.get(col, 0)
        if i > 0:
            p = ((c - i) / i) * 100
        else:
            p = 0
        table.append({"Metric": col.replace("Per Subscriber", "").strip(),
                      "Client Rate": c,
                      "Industry Avg Rate": i,
                      "Difference (%)": p})

    return {"CaseLoad_Diff": diff,
            "Util_Comparison": pd.DataFrame(table)}

# -----------------------------
# SENTIMENT COLORING
# -----------------------------
def get_sentiment(val, is_case_load=False):
    if is_case_load:
        if val < -10:
            return "Low (Good)", BENCHMARK_COLOR_GOOD
        if val > 10:
            return "High (Caution)", BENCHMARK_COLOR_BAD
        return "On Par", BRAND_COLOR_BLUE
    else:
        if val > 10:
            return "High (Good)", BENCHMARK_COLOR_GOOD
        if val < -10:
            return "Low (Caution)", BENCHMARK_COLOR_BAD
        return "On Par", BRAND_COLOR_BLUE

# -----------------------------
# CASE TYPE TABLE
# -----------------------------
def create_case_type_table(d):
    cols = [
        "Medical Cases", "Security Cases", "Travel Cases",
        "Medical Cases: I&A", "Medical Cases: Out-Patient",
        "Medical Cases: In-Patient", "Medical Cases: Evacuation / Repatriation, & RMR",
        "Security cases: I&A", "Security cases: Referral",
        "Security cases: Interventional Assistance", "Security cases: Evacuation",
        "Security cases: Active Monitoring"
    ]

    data = []
    for c in cols:
        data.append({"Case Type": c.replace("Cases:", "-"),
                     "Total Cases": d.get(c, 0)})

    df = pd.DataFrame(data)
    total = df["Total Cases"].sum()
    df["% of Total"] = (df["Total Cases"] / total * 100) if total > 0 else 0
    return df

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="International SOS Benchmark", layout="wide")
st.title("Travel Services Benchmarking and Forecast")

if RAW_DATA_DF.empty or BENCHMARK_DF.empty:
    st.error("Missing or unreadable CSV files.")
    st.stop()

# -----------------------------
# ACCOUNT SELECTION
# -----------------------------
acct_list = sorted(RAW_DATA_DF["AccountID"].unique().tolist())
acct = st.selectbox("Select Account ID", [""] + acct_list)

if acct:
    client = get_client_data(acct)
    bench = benchmark_client(client, BENCHMARK_DF)

    # HEADER FIELDS
    col1, col2, col3 = st.columns(3)
    col1.metric("Subscribers", f"{client.get('Subscribers', 0):,}")
    col2.metric("Industry", client.get("Clean_Industry", "Unknown"))

    since = client.get("Customer_Since")
    if pd.notna(since):
        yrs = (datetime.now() - since).days / 365.25
        t = f"{since.strftime('%Y-%m-%d')} ({yrs:.1f} yrs)"
    else:
        t = "N/A"
    col3.metric("Customer Since", t)

    st.subheader("Case Load Comparison")
    diff = bench["CaseLoad_Diff"]
    label, color = get_sentiment(diff, is_case_load=True)
    st.markdown(f"<h2 style='color:{color}'>{label}: {abs(diff):.1f}%</h2>", unsafe_allow_html=True)

    st.subheader("Case Breakdown")
    st.dataframe(create_case_type_table(client), hide_index=True, use_container_width=True)

    st.subheader("Utilization Comparison")
    st.dataframe(bench["Util_Comparison"], hide_index=True, use_container_width=True)

# -----------------------------
# PROJECTION MODEL
# -----------------------------
st.header("Case Projection Model")
inds = sorted(BENCHMARK_DF["Clean_Industry"].unique().tolist())
pi = st.selectbox("Industry", inds)
ps = st.number_input("Subscribers", min_value=1, value=1000)

row = BENCHMARK_DF[BENCHMARK_DF["Clean_Industry"] == pi].iloc[0]
case_cols = [c for c in BENCHMARK_DF.columns if "Per Subscriber" in c and c != "Total Cases Per Subscriber"]

projections = []
total_proj = 0
for c in case_cols:
    rate = row.get(c, 0)
    val = rate * ps
    total_proj += val
    projections.append({"Case Type": c.replace(" Per Subscriber", ""), "Projected Cases": val})

dfproj = pd.DataFrame(projections)
if total_proj > 0:
    dfproj["% of Total"] = dfproj["Projected Cases"] / total_proj * 100
else:
    dfproj["% of Total"] = 0

st.metric("Total Projected Cases", f"{total_proj:,.0f}")
st.dataframe(dfproj, hide_index=True, use_container_width=True)

# END
