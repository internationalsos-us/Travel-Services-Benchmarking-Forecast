import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# --- Global Configuration ---
BRAND_COLOR_BLUE = "#2f4696"
BRAND_COLOR_DARK = "#232762"
BENCHMARK_COLOR_GOOD = "#009354"
BENCHMARK_COLOR_BAD = "#D4002C"

# --- Column Definitions (CRITICAL: Derived from your CSV headers) ---
# Use these exact names to look up data in the DataFrames
RAW_ACCOUNT_COLS = ['AccountID', 'Business_Industry', 'Subscribers', 'Customer_Since']

RAW_CASE_COLS = [
    'Travel Cases', 'Medical Cases: I&A', 'Medical Cases: Out-Patient', 
    'Medical Cases: In-Patient', 'Medical Cases: Evacuation / Repatriation, & RMR', 
    'Security cases: I&A', 'Security cases: Referral', 'Security cases: Interventional Assistance', 
    'Security cases: Evacuation', 'Security cases: Active Monitoring'
]

RAW_UTIL_COLS = [
    'App & Portal Sessions', 'Alerts Sent to Travellers', 
    'Pre Trip Advisories Sent', 'DLP Completed Courses'
]

# --- Data Loading and Preprocessing ---

@st.cache_data
def load_data():
    try:
        raw_df = pd.read_csv("app_raw_data.csv")
        
        # --- CRITICAL CLEANUP: Strip whitespace and standardize headers ---
        raw_df.columns = raw_df.columns.str.strip()
        
        # Rename columns to standardized names for predictable internal access
        raw_df.rename(columns={
            'Business_Industry': 'Clean_Industry', 
            'Customer_Since': 'Customer_Since_Clean',
            'Total Cases': 'Total_Cases_Client_Raw' # The original total case count column
        }, inplace=True, errors='ignore')
        
        # Calculate TOTAL CASES for comparison and rates
        raw_df['Total_Cases_Calculated'] = raw_df[RAW_CASE_COLS].sum(axis=1)
        
        # Data Type Conversion
        raw_df['AccountID'] = raw_df['AccountID'].astype(str)
        raw_df['Customer_Since_Clean'] = pd.to_datetime(raw_df['Customer_Since_Clean'], errors='coerce')
        
        return raw_df
    except FileNotFoundError:
        st.error("Data file (app_raw_data.csv) not found. Please ensure it is uploaded.")
        return pd.DataFrame()

RAW_DATA_DF = load_data()


# --- Core Logic Functions ---

@st.cache_data
def calculate_industry_benchmarks(df):
    """Calculates all necessary industry average rates from the raw data."""
    if df.empty:
        return pd.DataFrame()
    
    # 1. Prepare aggregation dictionary
    agg_dict = {
        'Total_Subscribers': pd.NamedAgg(column='Subscribers', aggfunc='sum')
    }
    
    # Add all case and utilization columns to the aggregation dictionary
    all_count_cols = RAW_CASE_COLS + RAW_UTIL_COLS + ['Total_Cases_Calculated']
    for col in all_count_cols:
        agg_dict[col] = pd.NamedAgg(column=col, aggfunc='sum')
    
    # 2. Aggregate sums by Business Industry
    industry_agg = df.groupby('Clean_Industry').agg(**agg_dict).reset_index()
    
    # 3. Calculate "Rate per Subscriber" for Cases and Utilization
    for col in all_count_cols:
        # Avoid division by zero
        rate_col = f"{col} Per Subscriber"
        industry_agg[rate_col] = np.where(
            industry_agg['Total_Subscribers'] > 0, 
            industry_agg[col] / industry_agg['Total_Subscribers'], 
            0
        )
        
    return industry_agg

INDUSTRY_BENCHMARK_DF = calculate_industry_benchmarks(RAW_DATA_DF)


def get_client_data(account_id, df_raw):
    """Retrieves client data and calculates all necessary rates."""
    if df_raw.empty:
        return None
    
    client_row = df_raw[df_raw['AccountID'] == account_id].iloc[0]
    client_data = client_row.to_dict()
    client_data = {key.strip(): value for key, value in client_data.items()} # Clean dict keys
    
    subscribers = client_data.get('Subscribers', 0)
    
    # Define all base columns that need a 'Per Subscriber' rate calculation
    ALL_RATE_BASE_COLS = RAW_CASE_COLS + RAW_UTIL_COLS + ['Total_Cases_Calculated']
    
    # Calculate client's utilization and case rates per subscriber
    for original_col_name in ALL_RATE_BASE_COLS:
        rate_col_name = f"{original_col_name} Per Subscriber"
        raw_value = client_data.get(original_col_name, 0)
        
        if subscribers > 0 and raw_value != 0:
            client_data[rate_col_name] = raw_value / subscribers
        else:
            client_data[rate_col_name] = 0

    return client_data

def benchmark_client(client_data, industry_benchmark_df):
    """Performs client vs industry comparison."""
    
    industry = client_data['Clean_Industry'] 
    industry_row_match = industry_benchmark_df[industry_benchmark_df['Clean_Industry'] == industry]
    
    if industry_row_match.empty:
        return {'CaseLoad_Diff': 0, 'Util_Comparison': pd.DataFrame()}
        
    industry_row = industry_row_match.iloc[0]
    
    results = {}
    
    # 1. Case Load Benchmarking (Total Cases)
    total_cases_rate_client = client_data['Total_Cases_Calculated Per Subscriber']
    total_cases_rate_industry = industry_row['Total_Cases_Calculated Per Subscriber']
    
    if total_cases_rate_industry > 0:
        case_diff = ((total_cases_rate_client - total_cases_rate_industry) / total_cases_rate_industry) * 100
    else:
        case_diff = 0
    
    results['CaseLoad_Diff'] = case_diff
    
    # 2. Utilization Benchmarking (F, G, H, I)
    util_cols_map = {
        'App & Portal Sessions': 'App & Portal Sessions Per Subscriber',
        'Alerts Sent to Travellers': 'Alerts Sent to Travellers Per Subscriber',
        'Pre Trip Advisories Sent': 'Pre Trip Advisories Sent Per Subscriber',
        'DLP Completed Courses': 'DLP Completed Courses Per Subscriber'
    }
    
    util_comparison_table = []
    
    for metric_base_name, industry_rate_col in util_cols_map.items():
        client_rate_col = f"{metric_base_name} Per Subscriber"
        
        client_rate = client_data.get(client_rate_col, 0)
        industry_rate = industry_row[industry_rate_col]
        
        if industry_rate > 0:
            diff_percent = ((client_rate - industry_rate) / industry_rate) * 100
        else:
            diff_percent = 0
            
        util_comparison_table.append({
            'Metric': metric_base_name,
            'Client Rate': client_rate,
            'Industry Avg Rate': industry_rate,
            'Difference (%)': diff_percent
        })
        
    results['Util_Comparison'] = pd.DataFrame(util_comparison_table)
    
    return results

def get_sentiment(diff_percent, is_case_load=False):
    """Returns classification and color based on difference percentage."""
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
    """Creates a table for client's total cases by type."""
    
    case_type_display_map = {
        'Travel Cases': 'Travel Disruptions', 
        'Medical Cases: I&A': 'Medical - Information & Analysis', 
        'Medical Cases: Out-Patient': 'Medical - Out-Patient', 
        'Medical Cases: In-Patient': 'Medical - In-Patient', 
        'Medical Cases: Evacuation / Repatriation, & RMR': 'Medical - Evac/Repat/RMR',
        'Security cases: I&A': 'Security - Information & Analysis', 
        'Security cases: Referral': 'Security - Referrals', 
        'Security cases: Interventional Assistance': 'Security - Interventions', 
        'Security cases: Evacuation': 'Security - Evacuations', 
        'Security cases: Active Monitoring': 'Security - Active Monitoring'
    }
    
    table_data = []
    for raw_col, display_name in case_type_display_map.items():
        table_data.append({
            'Case Type': display_name, 
            'Total Cases': client_data.get(raw_col, 0)
        })
    
    df = pd.DataFrame(table_data)
    
    total = client_data.get('Total_Cases_Calculated', 0)
    
    df['% of Total'] = (df['Total Cases'] / total) * 100 if total > 0 else 0
    
    return df

# --- Page Setup ---

st.set_page_config(page_title="International SOS | Benchmarking", layout="wide")

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

# --- FIRST SECTION: Benchmarking ---
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">1. Account Benchmarking & Utilization Analysis</h2>', unsafe_allow_html=True)

if RAW_DATA_DF.empty:
    st.warning("Cannot run the app. Data file is required.")
    st.stop()

# 1.1 Input Fields
account_id_list = sorted(RAW_DATA_DF['AccountID'].astype(str).unique().tolist())
selected_account_id = st.selectbox("1.1 Account ID (Select to look up)", [''] + account_id_list)

if selected_account_id:
    client_data = get_client_data(selected_account_id, RAW_DATA_DF)
    benchmarks = benchmark_client(client_data, INDUSTRY_BENCHMARK_DF)
    
    # Retrieve auto-populated fields
    customer_since_date = client_data.get('Customer_Since_Clean', pd.NaT)
    customer_since_str = customer_since_date.strftime('%Y-%m-%d') if pd.notna(customer_since_date) else 'N/A'
    
    # 1.2 & 1.3 Auto-populated Fields
    col_sub, col_ind, col_since = st.columns(3)
    with col_sub:
        st.text_input("1.2 Number of Subscribers", value=f"{client_data['Subscribers']:,}", disabled=True)
    with col_ind:
        st.text_input("1.3 Business Industry", value=client_data['Clean_Industry'], disabled=True) # Use Clean_Industry
    with col_since:
        # Calculate Customer Since duration
        if pd.notna(customer_since_date):
            years_since = (datetime.now() - customer_since_date).days / 365.25
            since_text = f"{customer_since_str} ({years_since:.1f} years)"
        else:
            since_text = 'N/A'
        st.text_input("Customer Since", value=since_text, disabled=True)

    st.markdown('---')
    st.markdown(f'<h3 style="color:{BRAND_COLOR_DARK};">Case Activity Summary for Account ID: {selected_account_id}</h3>', unsafe_allow_html=True)
    
    # --- Case Load Comparison ---
    case_diff = benchmarks['CaseLoad_Diff']
    case_sentiment, case_color = get_sentiment(case_diff, is_case_load=True)
    
    col_metric, col_chart = st.columns([1, 2])
    
    with col_metric:
        st.markdown(f"""
        <div style="background-color: #f7f7f7; padding: 15px; border-radius: 8px; text-align: center;">
            <p style="font-size: 16px; margin-bottom: 0; color: grey;">Total Cases vs. Industry Average</p>
            <h1 style="color: {case_color}; font-size: 40px; margin: 0;">{case_sentiment}</h1>
            <p style="font-size: 20px; margin-top: 5px; color: {case_color};">
                {abs(case_diff):.1f}% {'Below' if case_diff < 0 else 'Above'} Avg.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        # Display client's total cases by case type
        case_df_for_display = create_case_type_table(client_data)
        st.subheader("Total Cases by Case Type")
        st.dataframe(
            case_df_for_display.style.format({'% of Total': '{:.1f}%', 'Total Cases': '{:,.0f}'}), 
            hide_index=True, 
            use_container_width=True
        )

    st.markdown('---')
    st.markdown(f'<h3 style="color:{BRAND_COLOR_DARK};">Utilization Analysis: Client vs. Industry Average</h3>', unsafe_allow_html=True)
    
    # --- Utilization Comparison ---
    util_df = benchmarks['Util_Comparison']
    
    def style_util_diff(val):
        sentiment, color = get_sentiment(val, is_case_load=False)
        return f'color: {color}; font-weight: bold; background-color: #f0fff0' if sentiment == 'High (Good)' else f'color: {color}'

    # Format and display utilization table
    st.dataframe(
        util_df.style.format({
            'Client Rate': '{:.3f}', 
            'Industry Avg Rate': '{:.3f}', 
            'Difference (%)': '{:+.1f}%'
        }).applymap(style_util_diff, subset=['Difference (%)']),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(f'<p><i>Utilization metrics are normalized by the number of subscribers. Analysis shows whether the client\'s utilization (App, Alerts, Advisories, DLP) is leading to more or less assistance cases compared to the industry average.</i></p>', unsafe_allow_html=True)
    
else:
    st.info("Please select an Account ID to view the benchmarking report.")

st.markdown('---')

# --- SECOND SECTION: Projection Model ---
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">2. Case Projection Model (Industry Average)</h2>', unsafe_allow_html=True)

if INDUSTRY_BENCHMARK_DF.empty:
    st.info("Cannot run projection. Data files are required.")
    st.stop()
    
# 2.1 & 2.2 Input Fields
col_proj_ind, col_proj_sub = st.columns(2)
with col_proj_ind:
    industry_list = sorted(INDUSTRY_BENCHMARK_DF['Clean_Industry'].unique().tolist())
    proj_industry = st.selectbox("2.1 Select Business Industry", industry_list)
with col_proj_sub:
    proj_subscribers = st.number_input("2.2 Enter Number of Subscribers", min_value=1, value=1000, step=1)

if proj_industry:
    proj_row = INDUSTRY_BENCHMARK_DF[INDUSTRY_BENCHMARK_DF['Clean_Industry'] == proj_industry].iloc[0]
    
    # Calculate projected cases
    projection_data = []
    total_projected_cases = 0
    
    # Base columns for projection model (using Industry rates per Subscriber, not per 100)
    # The rates were calculated in calculate_industry_benchmarks
    PROJECTION_RATE_COLS = [f"{col} Per Subscriber" for col in RAW_CASE_COLS]

    # Map raw rate column names to final display names
    PROJECTION_DISPLAY_MAP = {
        'Travel Cases Per Subscriber': 'Travel Disruptions', 
        'Medical Cases: I&A Per Subscriber': 'Medical - Information & Analysis', 
        'Medical Cases: Out-Patient Per Subscriber': 'Medical - Out-Patient', 
        'Medical Cases: In-Patient Per Subscriber': 'Medical - In-Patient', 
        'Medical Cases: Evacuation / Repatriation, & RMR Per Subscriber': 'Medical - Evac/Repat/RMR',
        'Security cases: I&A Per Subscriber': 'Security - Information & Analysis', 
        'Security cases: Referral Per Subscriber': 'Security - Referrals', 
        'Security cases: Interventional Assistance Per Subscriber': 'Security - Interventions', 
        'Security cases: Evacuation Per Subscriber': 'Security - Evacuations', 
        'Security cases: Active Monitoring Per Subscriber': 'Security - Active Monitoring'
    }
    
    for rate_col in PROJECTION_RATE_COLS:
        case_rate = proj_row.get(rate_col, 0)
        projected_cases = case_rate * proj_subscribers
        total_projected_cases += projected_cases
        
        # Determine the display name
        base_name = rate_col.replace(' Per Subscriber', '')
        display_name = PROJECTION_DISPLAY_MAP.get(base_name, base_name)
        
        projection_data.append({
            'Case Type': display_name,
            'Projected Cases (Last 12 Months)': projected_cases
        })

    proj_df = pd.DataFrame(projection_data)
    
    # Calculate percentage contribution
    proj_df['% of Total Projected Cases'] = (proj_df['Projected Cases (Last 12 Months)'] / total_projected_cases) * 100
    
    # Display Results
    st.markdown(f'<h3 style="color:{BRAND_COLOR_DARK};">Projected Annual Cases for {proj_industry}</h3>', unsafe_allow_html=True)
    
    col_total, col_empty = st.columns(2)
    with col_total:
         st.metric("Total Projected Cases", f"{total_projected_cases:,.0f}")
    
    st.dataframe(
        proj_df.style.format({
            'Projected Cases (Last 12 Months)': '{:,.0f}', 
            '% of Total Projected Cases': '{:.1f}%'
        }), 
        hide_index=True, 
        use_container_width=True
    )

st.markdown('---')

# --- Footer Sections ---
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">Value Proposition and Next Steps</h2>', unsafe_allow_html=True)
st.write("""
This section highlights the services that contribute to the measured ROI and outlines future risk mitigation strategies.
- **24/7 Support**: Immediate access to medical and security experts worldwide.
- **Risk Mitigation**: Proactive threat intelligence and risk forecasting.
- **Duty of Care**: Ensuring compliance with global standards.
""")

# Get in Touch Section
st.markdown("""
<div style="background-color:#232762; padding:40px; text-align:center; margin-top: 40px;">
    <h2 style="color:white;">How we can support</h2>
    <p style="color:white; font-size:16px; max-width:700px; margin:auto; margin-bottom:20px;">
    Protecting your people from health and security threats. 
    Our comprehensive Travel Risk Management program supports both managers and employees by proactively 
    identifying, alerting, and managing medical, security, mental wellbeing, and logistical risks.
    </p>
    <a href="https://www.internationalsos.com/get-in-touch?utm_source=benchmarkingreport" target="_blank">
        <button style="background-color:#EF820F; color:white; font-weight:bold; 
                       border:none; padding:15px 30px; font-size:16px; cursor:pointer; 
                       margin-top:15px; border-radius:20px;">
            Get in Touch
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown("""
<div style="text-align:center; font-size:12px; color:gray; margin-top:20px;">
© 2025 International SOS. WORLDWIDE REACH. HUMAN TOUCH.
</div>
""", unsafe_allow_html=True)
