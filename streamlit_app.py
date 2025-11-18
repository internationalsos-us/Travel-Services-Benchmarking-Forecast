import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# --- Global Configuration ---
BRAND_COLOR_BLUE = "#2f4696"
BRAND_COLOR_DARK = "#232762"
BENCHMARK_COLOR_GOOD = "#009354"
BENCHMARK_COLOR_BAD = "#D4002C"

# --- Column Mapping (Based on your specific instructions A-S) ---
# Keys are the CSV headers, Values are internal friendly names
COLUMN_MAP = {
    "Account_ID": "AccountID",                                      # A
    "Business_Industry": "Business_Industry",                       # B
    "WFR": "WFR",                                                   # C
    "Subscribers": "Subscribers",                                   # D
    "Customer_Since": "Customer_Since",                             # E
    "App_and_Portal_Sessions": "App_and_Portal_Sessions",           # F
    "Alerts_Sent_to_Travelers": "Alerts_Sent_to_Travelers",         # G
    "Pre_Trip_Advisories_Sent": "Pre_Trip_Advisories_Sent",         # H
    "E_Learning_Completed_Courses": "E_Learning_Completed_Courses", # I
    "Travel_Cases": "Travel_Cases",                                 # J
    "Medical_Cases_Information_and_Analysis": "Medical_Cases_IA",   # K
    "Medical_Cases_Out_Patient": "Medical_Cases_OutPatient",        # L
    "Medical_Cases_In_Patient": "Medical_Cases_InPatient",          # M
    "Medical_Cases_Evacuation_Repatriation_RMR": "Medical_Cases_Evac", # N
    "Security_Cases_Information_and_Analysis": "Security_Cases_IA",    # O
    "Security_Cases_Referrals": "Security_Cases_Referrals",            # P
    "Security_Cases_Interventional_Assistance": "Security_Cases_Intervention", # Q
    "Security_cases_Evacuation": "Security_Cases_Evac",                # R
    "Security_cases_Active_Monitoring": "Security_Cases_ActiveMonitoring" # S
}

# List of columns that represent CASES (J through S mapped names)
CASE_COLUMNS = [
    "Travel_Cases",
    "Medical_Cases_IA",
    "Medical_Cases_OutPatient",
    "Medical_Cases_InPatient",
    "Medical_Cases_Evac",
    "Security_Cases_IA",
    "Security_Cases_Referrals",
    "Security_Cases_Intervention",
    "Security_Cases_Evac",
    "Security_Cases_ActiveMonitoring"
]

# List of columns that represent UTILIZATION (F through I mapped names)
UTIL_COLUMNS = [
    "App_and_Portal_Sessions",
    "Alerts_Sent_to_Travelers",
    "Pre_Trip_Advisories_Sent",
    "E_Learning_Completed_Courses"
]

# --- Data Loading and Preprocessing ---

@st.cache_data
def load_data():
    try:
        # Load the single raw data file
        # NOTE: Ensure the file in GitHub is named 'app_raw_data.csv'
        df = pd.read_csv("app_raw_data.csv")
        
        # 1. Rename columns to match our internal mapping for consistency
        # Clean whitespace from CSV headers first
        df.columns = df.columns.str.strip()
        
        # Check if required columns exist (using the keys from your instruction)
        missing_cols = [col for col in COLUMN_MAP.keys() if col not in df.columns]
        if missing_cols:
            # If exact match fails, try a looser match (case insensitive, ignore underscore differences)
            # This handles slight variations in the CSV header text
            rename_dict = {}
            for required_col in COLUMN_MAP.keys():
                if required_col in df.columns:
                    rename_dict[required_col] = COLUMN_MAP[required_col]
                else:
                    # Try to find a close match in the actual columns
                    found = False
                    for actual_col in df.columns:
                        # Normalize strings for comparison: lowercase, remove spaces/underscores
                        req_norm = required_col.lower().replace('_', '').replace(' ', '')
                        act_norm = actual_col.lower().replace('_', '').replace(' ', '')
                        if req_norm == act_norm:
                            rename_dict[actual_col] = COLUMN_MAP[required_col]
                            found = True
                            break
                    if not found:
                        st.error(f"Critical Error: Cannot find column '{required_col}' in CSV.")
                        return pd.DataFrame()
            
            # Apply the smart rename
            df = df.rename(columns=rename_dict)
            
        else:
            # Exact match rename
            df = df.rename(columns=COLUMN_MAP)
        
        # 2. Data Type Conversion
        df['AccountID'] = df['AccountID'].astype(str)
        df['Customer_Since'] = pd.to_datetime(df['Customer_Since'], errors='coerce')
        
        # 3. Calculate Total Cases (Sum of all case columns)
        df['Total_Cases_Calculated'] = df[CASE_COLUMNS].sum(axis=1)
        
        return df
    except FileNotFoundError:
        st.error("Data file 'app_raw_data.csv' not found. Please upload it.")
        return pd.DataFrame()

RAW_DATA_DF = load_data()


# --- Core Logic Functions ---

@st.cache_data
def calculate_industry_averages(df):
    """
    Calculates the weighted average rates (per subscriber) for each industry.
    This creates the 'Benchmark' dataset dynamically from the raw data.
    """
    if df.empty:
        return pd.DataFrame()
        
    # Group by Industry
    industry_groups = df.groupby('Business_Industry')
    
    benchmarks = []
    
    for industry, group in industry_groups:
        total_subscribers = group['Subscribers'].sum()
        
        if total_subscribers == 0:
            continue
            
        # Calculate weighted average rate for each Case Type and Utilization metric
        # Formula: Sum of (Cases) / Sum of (Subscribers)
        
        industry_data = {
            'Business_Industry': industry,
            'Total_Subscribers_Base': total_subscribers
        }
        
        # Case Rates
        for col in CASE_COLUMNS:
            total_cases_in_industry = group[col].sum()
            industry_data[f"{col}_Rate"] = total_cases_in_industry / total_subscribers
            
        # Utilization Rates
        for col in UTIL_COLUMNS:
            total_util_in_industry = group[col].sum()
            industry_data[f"{col}_Rate"] = total_util_in_industry / total_subscribers
            
        # Total Cases Rate
        total_all_cases = group['Total_Cases_Calculated'].sum()
        industry_data['Total_Cases_Rate'] = total_all_cases / total_subscribers
        
        benchmarks.append(industry_data)
        
    return pd.DataFrame(benchmarks)

INDUSTRY_BENCHMARKS_DF = calculate_industry_averages(RAW_DATA_DF)


def get_client_metrics(account_id, raw_df, benchmark_df):
    """Retrieves client data and compares it to their industry benchmark."""
    if raw_df.empty:
        return None
        
    # 1. Get Client Row
    client_row = raw_df[raw_df['AccountID'] == account_id].iloc[0]
    client_industry = client_row['Business_Industry']
    client_subs = client_row['Subscribers']
    
    # 2. Get Industry Benchmark Row
    # Handle case where industry might not be in benchmark (e.g. singular data point)
    if client_industry in benchmark_df['Business_Industry'].values:
        industry_row = benchmark_df[benchmark_df['Business_Industry'] == client_industry].iloc[0]
    else:
        return None # Should not happen if benchmark is built from raw data
    
    metrics = {
        'Client_Name': str(client_row['AccountID']),
        'Industry': client_industry,
        'Subscribers': client_subs,
        'Customer_Since': client_row['Customer_Since'],
        'Benchmark_Results': {}
    }
    
    if client_subs > 0:
        # --- Case Load Comparison ---
        client_total_rate = client_row['Total_Cases_Calculated'] / client_subs
        ind_total_rate = industry_row['Total_Cases_Rate']
        
        if ind_total_rate > 0:
            diff = ((client_total_rate - ind_total_rate) / ind_total_rate) * 100
        else:
            diff = 0
            
        metrics['Case_Load_Diff'] = diff
        
        # --- Detailed Case Breakdown (Client Totals) ---
        metrics['Client_Case_Totals'] = {col: client_row[col] for col in CASE_COLUMNS}
        
        # --- Utilization Comparison ---
        util_comparison = []
        for col in UTIL_COLUMNS:
            c_rate = client_row[col] / client_subs
            i_rate = industry_row[f"{col}_Rate"]
            
            if i_rate > 0:
                u_diff = ((c_rate - i_rate) / i_rate) * 100
            else:
                u_diff = 0
            
            util_comparison.append({
                'Metric': col.replace('_', ' '),
                'Client Rate': c_rate,
                'Industry Avg': i_rate,
                'Difference (%)': u_diff
            })
        metrics['Util_Comparison'] = pd.DataFrame(util_comparison)
        
    return metrics

def run_projection(subscribers, industry, benchmark_df):
    """Calculates projected cases for a hypothetical number of subscribers."""
    if benchmark_df.empty:
        return pd.DataFrame(), 0
        
    industry_row = benchmark_df[benchmark_df['Business_Industry'] == industry].iloc[0]
    
    projections = []
    total_proj = 0
    
    for col in CASE_COLUMNS:
        rate = industry_row[f"{col}_Rate"]
        proj_count = rate * subscribers
        total_proj += proj_count
        
        # User friendly name mapping
        display_name = col.replace('Medical_Cases_', 'Medical - ').replace('Security_Cases_', 'Security - ').replace('_', ' ')
        
        projections.append({
            'Case Type': display_name,
            'Projected Cases': proj_count
        })
        
    df = pd.DataFrame(projections)
    if total_proj > 0:
        df['% of Total'] = (df['Projected Cases'] / total_proj) * 100
    else:
        df['% of Total'] = 0
        
    return df, total_proj

# --- Helper for Styling ---
def get_diff_color(val, invert=False):
    if invert: # For Cases (Lower is usually better/good, Higher is bad)
        if val < -10: return BENCHMARK_COLOR_GOOD
        if val > 10: return BENCHMARK_COLOR_BAD
    else: # For Utilization (Higher is usually better/good)
        if val > 10: return BENCHMARK_COLOR_GOOD
        if val < -10: return BENCHMARK_COLOR_BAD
    return BRAND_COLOR_BLUE

# --- Main App Layout ---

st.set_page_config(page_title="International SOS | Benchmarking", layout="wide")

# Banner
st.markdown(f"""
<div style="background-color:{BRAND_COLOR_DARK}; padding:20px; display:flex; align-items:center;">
    <img src="https://images.learn.internationalsos.com/EloquaImages/clients/InternationalSOS/%7B0769a7db-dae2-4ced-add6-d1a73cb775d5%7D_International_SOS_white_hr_%281%29.png" style="height:50px; margin-right:20px;">
    <h1 style="color:white; margin:0; font-size:24px;">Travel Services Benchmarking and Forecast</h1>
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown(f'<h1 style="color:{BRAND_COLOR_DARK};">Assistance Activity and ROI Benchmark</h1>', unsafe_allow_html=True)
st.write("Compare client assistance activity against industry peers (calculated dynamically from raw data).")
st.markdown('---')

# --- SECTION 1: Benchmarking ---
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">1. Account Benchmarking & Utilization Analysis</h2>', unsafe_allow_html=True)

if RAW_DATA_DF.empty:
    st.stop()

# 1.1 Inputs
ids = sorted(RAW_DATA_DF['AccountID'].unique())
selected_id = st.selectbox("1.1 Select Account ID", ids)

if selected_id:
    metrics = get_client_metrics(selected_id, RAW_DATA_DF, INDUSTRY_BENCHMARKS_DF)
    
    if metrics:
        # 1.2/1.3 Display
        c1, c2, c3 = st.columns(3)
        c1.text_input("1.2 Business Industry", value=metrics['Industry'], disabled=True)
        c2.text_input("1.3 Number of Subscribers", value=f"{metrics['Subscribers']:,}", disabled=True)
        
        cust_since = metrics['Customer_Since']
        since_str = cust_since.strftime('%Y-%m-%d') if pd.notna(cust_since) else "N/A"
        c3.text_input("Customer Since", value=since_str, disabled=True)
        
        st.markdown('---')
        
        # Summary
        diff = metrics.get('Case_Load_Diff', 0)
        color = get_diff_color(diff, invert=True) # Negative diff is Good for cases
        
        c_metric, c_chart = st.columns([1, 2])
        
        with c_metric:
            st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
                <h3 style="margin:0; color:{BRAND_COLOR_DARK}">Total Case Load</h3>
                <h1 style="font-size:48px; margin:10px 0; color:{color}">{abs(diff):.1f}%</h1>
                <p style="font-size:18px; font-weight:bold; color:{color}">
                    {'BELOW' if diff < 0 else 'ABOVE'} Industry Avg
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with c_chart:
            # Create Breakdown DF
            breakdown_data = []
            if 'Client_Case_Totals' in metrics:
                for col, val in metrics['Client_Case_Totals'].items():
                    name = col.replace('Medical_Cases_', 'Med - ').replace('Security_Cases_', 'Sec - ').replace('_', ' ')
                    breakdown_data.append({'Case Type': name, 'Cases': val})
            
            bd_df = pd.DataFrame(breakdown_data)
            if not bd_df.empty:
                total_c = bd_df['Cases'].sum()
                bd_df['%'] = (bd_df['Cases'] / total_c) * 100 if total_c > 0 else 0
                
                st.subheader("Client Case Breakdown")
                st.dataframe(bd_df.style.format({'%': '{:.1f}%'}), use_container_width=True, hide_index=True)

        st.write("")
        st.markdown(f'<h3 style="color:{BRAND_COLOR_DARK};">Utilization Analysis</h3>', unsafe_allow_html=True)
        
        if 'Util_Comparison' in metrics:
            util_df = metrics['Util_Comparison']
            
            def style_util(val):
                color = get_diff_color(val, invert=False) # Positive is Good for utilization
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                util_df.style.format({
                    'Client Rate': '{:.4f}',
                    'Industry Avg': '{:.4f}',
                    'Difference (%)': '{:+.1f}%'
                }).applymap(style_util, subset=['Difference (%)']),
                use_container_width=True,
                hide_index=True
            )

st.markdown('---')

# --- SECTION 2: Projection ---
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">2. Case Projection Model</h2>', unsafe_allow_html=True)

p1, p2 = st.columns(2)
with p1:
    if not INDUSTRY_BENCHMARKS_DF.empty:
        industries = sorted(INDUSTRY_BENCHMARKS_DF['Business_Industry'].unique())
        # Default to selected client's industry if available
        def_ix = 0
        if selected_id and metrics and metrics['Industry'] in industries:
             def_ix = industries.index(metrics['Industry'])
        proj_ind = st.selectbox("2.1 Select Industry", industries, index=def_ix)
    else:
        proj_ind = None
        st.warning("No industry data available.")

with p2:
    # Default to selected client's subs if available
    def_sub = int(metrics['Subscribers']) if selected_id and metrics else 1000
    proj_sub = st.number_input("2.2 Subscribers", min_value=1, value=def_sub)

if proj_ind and proj_sub and not INDUSTRY_BENCHMARKS_DF.empty:
    proj_df, total_p = run_projection(proj_sub, proj_ind, INDUSTRY_BENCHMARKS_DF)
    
    c_p_metric, c_p_table = st.columns([1, 2])
    with c_p_metric:
        st.metric("Projected Total Cases (Annual)", f"{total_p:,.1f}")
    
    with c_p_table:
        st.dataframe(
            proj_df.style.format({'Projected Cases': '{:.1f}', '% of Total': '{:.1f}%'}),
            use_container_width=True,
            hide_index=True
        )

# --- Footer ---
st.markdown('---')
st.markdown(f"""
<div style="text-align:center; color:gray; padding:20px;">
    <h3 style="color:{BRAND_COLOR_BLUE}">Value Proposition</h3>
    <p>Highlights services contributing to ROI: 24/7 Support, Risk Mitigation, Duty of Care.</p>
    <br>
    <a href="https://www.internationalsos.com/get-in-touch?utm_source=benchmarkingreport" target="_blank">
        <button style="background-color:{BRAND_COLOR_ORANGE}; color:white; border:none; padding:10px 20px; border-radius:5px; cursor:pointer;">
            Get in Touch
        </button>
    </a>
    <p style="margin-top:20px;">© 2025 International SOS. WORLDWIDE REACH. HUMAN TOUCH.</p>
</div>
""", unsafe_allow_html=True)
