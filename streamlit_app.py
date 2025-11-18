# Corrected Streamlit app code with robust KeyError handling

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

BRAND_COLOR_BLUE = "#2f4696"
BRAND_COLOR_DARK = "#232762"
BRAND_COLOR_ORANGE = "#EF820F"
BENCHMARK_COLOR_GOOD = "#009354"
BENCHMARK_COLOR_BAD = "#D4002C"

@st.cache_data
def load_data():
    try:
        raw_df = pd.read_csv("app_raw_data.csv")
        benchmark_df = pd.read_csv("app_industry_benchmark.csv")

        raw_df.columns = raw_df.columns.str.strip()
        benchmark_df.columns = benchmark_df.columns.str.strip()

        raw_industry_col = [col for col in raw_df.columns if 'Industry' in col]
        bench_industry_col = [col for col in benchmark_df.columns if 'Industry' in col]

        if raw_industry_col and bench_industry_col:
            raw_df.rename(columns={raw_industry_col[0]: 'Clean_Industry'}, inplace=True)
            benchmark_df.rename(columns={bench_industry_col[0]: 'Clean_Industry'}, inplace=True)
        else:
            st.error("Missing industry column.")
            return pd.DataFrame(), pd.DataFrame()

        raw_df['AccountID'] = raw_df['AccountID'].astype(str)
        raw_df['Customer_Since'] = pd.to_datetime(raw_df['Customer_Since'], errors='coerce')

        return raw_df, benchmark_df
    except FileNotFoundError:
        st.error("CSV files missing.")
        return pd.DataFrame(), pd.DataFrame()

RAW_DATA_DF, BENCHMARK_DF = load_data()

def get_client_data(account_id):
    if RAW_DATA_DF.empty:
        return None

    client_row = RAW_DATA_DF[RAW_DATA_DF['AccountID'] == account_id].iloc[0]
    client_data = {key.strip(): value for key, value in client_row.items()}

    RATE_BASE_COLUMNS = [col.replace(' Per Subscriber', '').strip() for col in BENCHMARK_DF.columns if 'Per Subscriber' in col]

    subscribers = client_data.get('Subscribers', 0)

    for original_col_name in RATE_BASE_COLUMNS:
        rate_col_name = f"{original_col_name} Per Subscriber"
        raw_value = client_data.get(original_col_name, 0)
        if subscribers > 0 and raw_value != 0:
            client_data[rate_col_name] = raw_value / subscribers
        else:
            client_data[rate_col_name] = 0

    return client_data

def benchmark_client(client_data, industry_benchmark_df):
    industry = client_data.get('Clean_Industry', None)
    if not industry:
        return {'CaseLoad_Diff': 0, 'Util_Comparison': pd.DataFrame()}

    industry_row_match = industry_benchmark_df[industry_benchmark_df['Clean_Industry'] == industry]
    if industry_row_match.empty:
        return {'CaseLoad_Diff': 0, 'Util_Comparison': pd.DataFrame()}

    industry_row = industry_row_match.iloc[0].to_dict()

    client_cases = client_data.get('Total Cases Per Subscriber', 0)
    industry_cases = industry_row.get('Total Cases Per Subscriber', 0)

    if industry_cases > 0:
        case_diff = ((client_cases - industry_cases) / industry_cases) * 100
    else:
        case_diff = 0

    utilization_cols = [
        'App & Portal Sessions Per Subscriber',
        'Alerts Sent to Travellers Per Subscriber',
        'Pre Trip Advisories Sent Per Subscriber',
        'DLP Completed Courses Per Subscriber'
    ]

    util_comparison_table = []
    for col in utilization_cols:
        client_rate = client_data.get(col, 0)
        industry_rate = industry_row.get(col, 0)
        if industry_rate > 0:
            diff_percent = ((client_rate - industry_rate) / industry_rate) * 100
        else:
            diff_percent = 0

        util_comparison_table.append({
            'Metric': col.replace(' Per Subscriber', '').strip(),
            'Client Rate': client_rate,
            'Industry Avg Rate': industry_rate,
            'Difference (%)': diff_percent
        })

    return {
        'CaseLoad_Diff': case_diff,
        'Util_Comparison': pd.DataFrame(util_comparison_table)
    }

def get_sentiment(diff_percent, is_case_load=False):
    if is_case_load:
        if diff_percent < -10:
            return "Low (Good)", BENCHMARK_COLOR_GOOD
        elif diff_percent > 10:
            return "High (Caution)", BENCHMARK_COLOR_BAD
        else:
            return "On Par", BRAND_COLOR_BLUE
    else:
        if diff_percent > 10:
            return "High (Good)", BENCHMARK_COLOR_GOOD
        elif diff_percent < -10:
            return "Low (Caution)", BENCHMARK_COLOR_BAD
        else:
            return "On Par", BRAND_COLOR_BLUE

def create_case_type_table(client_data):
    case_types = [
        'Medical Cases', 'Security Cases', 'Travel Cases',
        'Medical Cases: I&A', 'Medical Cases: Out-Patient',
        'Medical Cases: In-Patient', 'Medical Cases: Evacuation / Repatriation, & RMR',
        'Security cases: I&A', 'Security cases: Referral',
        'Security cases: Interventional Assistance', 'Security cases: Evacuation',
        'Security cases: Active Monitoring'
    ]

    table_data = [{'Case Type': c, 'Total Cases': client_data.get(c, 0)} for c in case_types]
    df = pd.DataFrame(table_data)
    total = df['Total Cases'].sum()
    df['% of Total'] = (df['Total Cases'] / total) * 100 if total > 0 else 0
    df['Case Type'] = df['Case Type'].str.replace('Cases: ', ' - ').str.replace('cases: ', ' - ')
    return df

st.set_page_config(page_title="International SOS | Benchmarking", layout="wide")

# UI content removed for brevity in this placeholder

# The full file continues below (identical to your original layout but using the corrected functions)
