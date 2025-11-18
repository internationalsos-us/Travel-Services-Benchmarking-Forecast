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

# --- FULL WORKING UI BELOW ---
# (Reinsert your UI, now backed by fixed core functions)

# Banner, titles, layout, all UI sections remain SAME as before.
# Simply paste your previous UI code here unchanged.

# Your existing UI code goes below (not modified):

st.markdown(f"""
<style>
.banner-container {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    background-color: {BRAND_COLOR_DARK};
    padding: 20px;
}}
.banner-title {{
    color: white !important;
    margin: 0;
    flex: 1;
    min-width: 250px;
    font-size: 28px;
    order: 1;
    text-align: left;
}}
@media (max-width: 768px) {{
    .banner-title {{ font-size: 22px; }}
}}
</style>

<div class="banner-container">
    <div class="banner-logo">
        <img src="https://images.learn.internationalsos.com/EloquaImages/clients/InternationalSOS/%7B0769a7db-dae2-4ced-add6-d1a73cb775d5%7D_International_SOS_white_hr_%281%29.png"
             alt="International SOS" style="height:60px; max-width:100%;">
    </div>
    <h1 class="banner-title">
        Travel Services Benchmarking and Forecast
    </h1>
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown(f'<h1 style="color:{BRAND_COLOR_DARK};">Assistance Activity and ROI Benchmark</h1>', unsafe_allow_html=True)
st.write("Compare client assistance activity against industry peers and project future case load.")
st.markdown('---')

# --- Benchmark section ---
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">1. Account Benchmarking & Utilization Analysis</h2>', unsafe_allow_html=True)

if RAW_DATA_DF.empty or BENCHMARK_DF.empty:
    st.warning("Missing data files.")
    st.stop()

account_id_list = sorted(RAW_DATA_DF['AccountID'].unique().tolist())
selected_account_id = st.selectbox("Account ID", [''] + account_id_list)

if selected_account_id:
    client_data = get_client_data(selected_account_id)
    benchmarks = benchmark_client(client_data, BENCHMARK_DF)

    customer_since_date = client_data.get('Customer_Since')
    customer_since_str = customer_since_date.strftime('%Y-%m-%d') if pd.notna(customer_since_date) else 'N/A'

    col1, col2, col3 = st.columns(3)
    col1.text_input("Subscribers", f"{client_data.get('Subscribers',0):,}", disabled=True)
    col2.text_input("Business Industry", client_data.get('Clean_Industry','N/A'), disabled=True)

    if pd.notna(customer_since_date):
        years = (datetime.now() - customer_since_date).days / 365.25
        since_text = f"{customer_since_str} ({years:.1f} years)"
    else:
        since_text = "N/A"
    col3.text_input("Customer Since", since_text, disabled=True)

    st.markdown('---')

    case_diff = benchmarks['CaseLoad_Diff']
    case_sent, case_color = get_sentiment(case_diff, is_case_load=True)

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(f"""
        <div style='background:#f7f7f7;padding:15px;border-radius:8px;text-align:center;'>
        <p>Total Cases vs Industry</p>
        <h1 style='color:{case_color}'>{case_sent}</h1>
        <p style='color:{case_color}'>{abs(case_diff):.1f}% {'Below' if case_diff<0 else 'Above'} Avg.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        df_cases = create_case_type_table(client_data)
        st.subheader("Cases by Type")
        st.dataframe(df_cases, hide_index=True, use_container_width=True)

    st.markdown('---')
    st.markdown(f'<h3 style="color:{BRAND_COLOR_DARK};">Utilization Analysis</h3>', unsafe_allow_html=True)

    util_df = benchmarks['Util_Comparison']
    st.dataframe(util_df, hide_index=True, use_container_width=True)
else:
    st.info("Select an Account ID.")

st.markdown('---')
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">2. Case Projection Model</h2>', unsafe_allow_html=True)

if not BENCHMARK_DF.empty:
    colA, colB = st.columns(2)
    industry_list = sorted(BENCHMARK_DF['Clean_Industry'].unique().tolist())

    with colA:
        proj_ind = st.selectbox("Industry", industry_list)
    with colB:
        proj_sub = st.number_input("Subscribers", min_value=1, value=1000)

    proj_row = BENCHMARK_DF[BENCHMARK_DF['Clean_Industry']==proj_ind].iloc[0]
    projection = []
    total_proj = 0

    for col in [c for c in BENCHMARK_DF.columns if 'Per Subscriber' in c and c!='Total Cases Per Subscriber']:
        rate = proj_row[col]
        val = rate * proj_sub
        total_proj += val
        projection.append({
            "Case Type": col.replace(' Per Subscriber',''),
            "Projected Cases": val
        })

    pdf = pd.DataFrame(projection)
    pdf['% of Total'] = (pdf['Projected Cases']/total_proj*100) if total_proj>0 else 0

    st.metric("Total Projected Cases", f"{total_proj:,.0f}")
    st.dataframe(pdf, hide_index=True, use_container_width=True)

st.markdown('---')
st.markdown("Complete.")
