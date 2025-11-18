import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# --- Global Configuration ---
# Using International SOS brand colors for consistency
BRAND_COLOR_BLUE = "#2f4696"
BRAND_COLOR_DARK = "#232762"
BRAND_COLOR_ORANGE = "#EF820F"
BENCHMARK_COLOR_GOOD = "#009354"
BENCHMARK_COLOR_BAD = "#D4002C"

# --- Data Loading and Preprocessing ---

@st.cache_data
def load_data():
    try:
        # Load the pre-processed data files (uploaded by user)
        raw_df = pd.read_csv("app_raw_data.csv")
        benchmark_df = pd.read_csv("app_industry_benchmark.csv")
        
        # --- EXPLICIT COLUMN CLEANUP AND VERIFICATION ---
        # 1. Standardize column names by stripping whitespace
        raw_df.columns = raw_df.columns.str.strip()
        benchmark_df.columns = benchmark_df.columns.str.strip()
        
        # 2. Force rename critical columns to simple, guaranteed names in the RAW data
        raw_df.rename(columns={
            # Target the problematic column names explicitly
            'Total Cases': 'Total_Cases_Clean',
            'Customer_Since': 'Customer_Since_Clean'
        }, inplace=True, errors='ignore')
        
        # 3. Identify and rename the Business Industry column to a guaranteed clean name ('Clean_Industry')
        raw_industry_col = [col for col in raw_df.columns if 'Industry' in col]
        bench_industry_col = [col for col in benchmark_df.columns if 'Industry' in col]
        
        if raw_industry_col and bench_industry_col:
            # Force rename both DataFrames to use the guaranteed clean name
            raw_df.rename(columns={raw_industry_col[0]: 'Clean_Industry'}, inplace=True)
            benchmark_df.rename(columns={bench_industry_col[0]: 'Clean_Industry'}, inplace=True)
        else:
            st.error("Data structure error: Cannot locate 'Business Industry' column even after robust search.")
            return pd.DataFrame(), pd.DataFrame()
        
        # 4. Final data checks
        if 'Clean_Industry' not in raw_df.columns or 'Clean_Industry' not in benchmark_df.columns:
             return pd.DataFrame(), pd.DataFrame()
        
        # Ensure AccountID is string for stable selectbox keying
        raw_df['AccountID'] = raw_df['AccountID'].astype(str)
        
        # Convert Customer_Since to datetime
        raw_df['Customer_Since_Clean'] = pd.to_datetime(raw_df['Customer_Since_Clean'], errors='coerce')
        
        return raw_df, benchmark_df
    except FileNotFoundError:
        st.error("Data files (app_raw_data.csv and app_industry_benchmark.csv) not found. Please ensure they are uploaded.")
        return pd.DataFrame(), pd.DataFrame()

RAW_DATA_DF, BENCHMARK_DF = load_data()

# --- Utility Functions ---

def get_client_data(account_id):
    if RAW_DATA_DF.empty:
        return None
    
    # Filter the raw data for the selected AccountID
    client_row = RAW_DATA_DF[RAW_DATA_DF['AccountID'] == account_id].iloc[0]
    
    # Convert the row to a dictionary, ensuring all keys (column names) are clean strings
    client_data = {key.strip(): value for key, value in client_row.items()}
    
    # Get the list of all base columns that need a 'Per Subscriber' rate calculation
    RATE_BASE_COLUMNS = [col.replace(' Per Subscriber', '').strip() 
                         for col in BENCHMARK_DF.columns if 'Per Subscriber' in col]
    
    subscribers = client_data.get('Subscribers', 0)
    
    # Calculate client's utilization and case rates per subscriber
    for original_col_name in RATE_BASE_COLUMNS:
        rate_col_name = f"{original_col_name} Per Subscriber"
        
        # --- FIX: Use the CLEANED names for the base columns if they were successfully renamed ---
        # Note: The 'Total Cases' column name in the raw data is now 'Total_Cases_Clean'
        
        if original_col_name == 'Total Cases':
            raw_value = client_data.get('Total_Cases_Clean', 0)
        elif original_col_name in client_data:
            raw_value = client_data.get(original_col_name, 0)
        else:
            raw_value = 0
        
        # Ensure the final rate column name is clean (e.g., 'Total Cases Per Subscriber')
        if subscribers > 0 and raw_value != 0:
            client_data[rate_col_name] = raw_value / subscribers
        else:
            client_data[rate_col_name] = 0 # Default to 0 if data is missing or no subscribers

    return client_data

def benchmark_client(client_data, industry_benchmark_df):
    # Compares client's utilization and case rates against industry average.
    
    industry = client_data['Clean_Industry'] # Use the clean name
    # Use a boolean mask to find the matching industry row safely
    industry_row_match = industry_benchmark_df[industry_benchmark_df['Clean_Industry'] == industry]
    
    if industry_row_match.empty:
        return {'CaseLoad_Diff': 0, 'Util_Comparison': pd.DataFrame()}
        
    industry_row = industry_row_match.iloc[0]
    
    results = {}
    
    # 1. Case Load Benchmarking (Access is now safe)
    total_cases_rate_client = client_data['Total Cases Per Subscriber']
    total_cases_rate_industry = industry_row['Total Cases Per Subscriber']
    
    if total_cases_rate_industry > 0:
        case_diff = ((total_cases_rate_client - total_cases_rate_industry) / total_cases_rate_industry) * 100
    else:
        case_diff = 0
    
    results['CaseLoad_Diff'] = case_diff
    
    # 2. Utilization Benchmarking (F, G, H, I)
    utilization_cols = [
        'App & Portal Sessions Per Subscriber', 
        'Alerts Sent to Travellers Per Subscriber', 
        'Pre Trip Advisories Sent Per Subscriber', 
        'DLP Completed Courses Per Subscriber'
    ]
    
    util_comparison_table = []
    
    for col in utilization_cols:
        # Use .get() for safe dictionary lookup
        client_rate = client_data.get(col, 0)
        industry_rate = industry_row[col]
        
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
        
    results['Util_Comparison'] = pd.DataFrame(util_comparison_table)
    
    return results

def get_sentiment(diff_percent, is_case_load=False):
    # Returns classification and color based on difference percentage.
    if is_case_load:
        # Case Load: Negative is good (lower than average)
        if diff_percent < -10:
            return "Low (Good)", BENCHMARK_COLOR_GOOD
        elif diff_percent > 10:
            return "High (Caution)", BENCHMARK_COLOR_BAD
        else:
            return "On Par", BRAND_COLOR_BLUE
    else:
        # Utilization: Positive is good (higher than average)
        if diff_percent > 10:
            return "High (Good)", BENCHMARK_COLOR_GOOD
        elif diff_percent < -10:
            return "Low (Caution)", BENCHMARK_COLOR_BAD
        else:
            return "On Par", BRAND_COLOR_BLUE

def create_case_type_table(client_data):
    # Creates a table for client's total cases by type.
    
    # List of all case columns used in the display
    case_types = [
        'Medical Cases', 'Security Cases', 'Travel Cases', 
        'Medical Cases: I&A', 'Medical Cases: Out-Patient', 
        'Medical Cases: In-Patient', 'Medical Cases: Evacuation / Repatriation, & RMR',
        'Security cases: I&A', 'Security cases: Referral', 
        'Security cases: Interventional Assistance', 'Security cases: Evacuation',
        'Security cases: Active Monitoring'
    ]
    
    # We retrieve the values from the client_data dictionary, using the cleaned names for base columns
    table_data = [{'Case Type': c, 'Total Cases': client_data.get(c, 0)} for c in case_types]
    
    # We must explicitly look up the Total Cases value using the CLEANED name for calculation
    total = client_data.get('Total_Cases_Clean', 0)
    
    df = pd.DataFrame(table_data)
    
    # Calculate percentage contribution
    df['% of Total'] = (df['Total Cases'] / total) * 100 if total > 0 else 0
    
    # Clean up names for display
    df['Case Type'] = df['Case Type'].str.replace('Cases: ', ' - ').str.replace('cases: ', ' - ')
    
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

if RAW_DATA_DF.empty or BENCHMARK_DF.empty:
    st.warning("Cannot run the app. Data files are required.")
    st.stop()

# 1.1 Input Fields
account_id_list = sorted(RAW_DATA_DF['AccountID'].astype(str).unique().tolist())
selected_account_id = st.selectbox("1.1 Account ID (Select to look up)", [''] + account_id_list)

if selected_account_id:
    client_data = get_client_data(selected_account_id)
    benchmarks = benchmark_client(client_data, BENCHMARK_DF)
    
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
        # Note: Streamlit styling requires double braces for formatting strings
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

if not BENCHMARK_DF.empty:
    
    # 2.1 & 2.2 Input Fields
    col_proj_ind, col_proj_sub = st.columns(2)
    with col_proj_ind:
        # Use Clean_Industry for safe lookup
        industry_list = sorted(BENCHMARK_DF['Clean_Industry'].unique().tolist())
        proj_industry = st.selectbox("2.1 Select Business Industry", industry_list)
    with col_proj_sub:
        proj_subscribers = st.number_input("2.2 Enter Number of Subscribers", min_value=1, value=1000, step=1)
    
    if proj_industry:
        # Use Clean_Industry for safe filtering
        proj_row = BENCHMARK_DF[BENCHMARK_DF['Clean_Industry'] == proj_industry].iloc[0]
        
        # Calculate projected cases
        projection_data = []
        total_projected_cases = 0
        
        # Iterate over all detailed case rate columns
        case_rate_cols = [col for col in BENCHMARK_DF.columns if 'Per Subscriber' in col and col != 'Total Cases Per Subscriber']
        
        for col in case_rate_cols:
            case_rate = proj_row[col]
            projected_cases = case_rate * proj_subscribers
            total_projected_cases += projected_cases
            
            projection_data.append({
                'Case Type': col.replace(' Per Subscriber', '').replace('Cases: ', ' - ').replace('cases: ', ' - '),
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
else:
    st.info("Cannot run projection. Data files are required.")

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
