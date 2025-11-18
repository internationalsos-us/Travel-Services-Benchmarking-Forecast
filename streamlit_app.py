import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# --- Global Configuration ---
BRAND_COLOR_BLUE = "#2f4696"
BRAND_COLOR_DARK = "#232762"
BENCHMARK_COLOR_GOOD = "#009354"
BENCHMARK_COLOR_BAD = "#D4002C"
BRAND_COLOR_ORANGE = "#EF820F"

# --- Column Mapping ---
# This maps the CSV headers from your file to internal friendly variable names
COLUMN_MAP = {
    "Account_ID": "AccountID",
    "Business_Industry": "Business_Industry",
    "WFR": "WFR",
    "Subscribers": "Subscribers",
    "Customer_Since": "Customer_Since",
    "App_and_Portal_Sessions": "App_and_Portal_Sessions",
    "Alerts_Sent_to_Travelers": "Alerts_Sent_to_Travelers",
    "Pre_Trip_Advisories_Sent": "Pre_Trip_Advisories_Sent",
    "E_Learning_Completed_Courses": "E_Learning_Completed_Courses",
    "Travel_Cases": "Travel_Cases",
    "Medical_Cases_Information_and_Analysis": "Medical_Cases_IA",
    "Medical_Cases_Out_Patient": "Medical_Cases_OutPatient",
    "Medical_Cases_In_Patient": "Medical_Cases_InPatient",
    "Medical_Cases_Evacuation_Repatriation_RMR": "Medical_Cases_Evac",
    "Security_Cases_Information_and_Analysis": "Security_Cases_IA",
    "Security_Cases_Referrals": "Security_Cases_Referrals",
    "Security_Cases_Interventional_Assistance": "Security_Cases_Intervention",
    "Security_cases_Evacuation": "Security_Cases_Evac",
    "Security_cases_Active_Monitoring": "Security_Cases_ActiveMonitoring"
}

# Columns for CASE Metrics (J-S)
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

# Columns for UTILIZATION Metrics (F-I)
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
        # Note: Ensure the file in GitHub is named 'app_raw_data.csv'
        df = pd.read_csv("app_raw_data.csv")
        
        # Clean headers: remove whitespace
        df.columns = df.columns.str.strip()
        
        # Rename columns using our explicit map
        # If a column is missing, we will report it, but try to proceed if critical ones exist
        rename_dict = {}
        for csv_header, internal_name in COLUMN_MAP.items():
            if csv_header in df.columns:
                rename_dict[csv_header] = internal_name
            else:
                # Fallback: try case-insensitive match if exact match fails
                found = False
                for col in df.columns:
                    if col.lower().replace(' ', '').replace('_', '') == csv_header.lower().replace(' ', '').replace('_', ''):
                        rename_dict[col] = internal_name
                        found = True
                        break
                if not found:
                    st.error(f"Critical Error: Column '{csv_header}' not found in CSV.")
                    return pd.DataFrame()

        df = df.rename(columns=rename_dict)
        
        # Data conversions
        df['AccountID'] = df['AccountID'].astype(str)
        df['Customer_Since'] = pd.to_datetime(df['Customer_Since'], errors='coerce')
        
        # Calculate Total Cases (Sum of all specific case types)
        df['Total_Cases_Calculated'] = df[CASE_COLUMNS].sum(axis=1)
        
        return df
    except FileNotFoundError:
        st.error("Data file 'app_raw_data.csv' not found. Please upload it.")
        return pd.DataFrame()

RAW_DATA_DF = load_data()


# --- Logic: Industry Benchmarks (Dynamic Calculation) ---

@st.cache_data
def calculate_industry_averages(df):
    """
    Dynamically calculates industry averages (per subscriber) from the raw data.
    """
    if df.empty:
        return pd.DataFrame()
        
    industry_groups = df.groupby('Business_Industry')
    benchmarks = []
    
    for industry, group in industry_groups:
        total_subscribers = group['Subscribers'].sum()
        
        if total_subscribers == 0:
            continue
            
        industry_data = {
            'Business_Industry': industry,
            'Total_Subscribers_Base': total_subscribers
        }
        
        # Calculate average rate per subscriber for CASES
        for col in CASE_COLUMNS:
            total_cases_in_industry = group[col].sum()
            industry_data[f"{col}_Rate"] = total_cases_in_industry / total_subscribers
            
        # Calculate average rate per subscriber for UTILIZATION
        for col in UTIL_COLUMNS:
            total_util_in_industry = group[col].sum()
            industry_data[f"{col}_Rate"] = total_util_in_industry / total_subscribers
            
        # Total Cases Rate
        total_all_cases = group['Total_Cases_Calculated'].sum()
        industry_data['Total_Cases_Rate'] = total_all_cases / total_subscribers
        
        benchmarks.append(industry_data)
        
    return pd.DataFrame(benchmarks)

INDUSTRY_BENCHMARKS_DF = calculate_industry_averages(RAW_DATA_DF)


# --- Logic: Client Metrics & Comparison ---

def get_client_metrics(account_id, raw_df, benchmark_df):
    """Retrieves client data and compares against industry benchmarks."""
    if raw_df.empty:
        return None
        
    # Get Client Data
    client_row = raw_df[raw_df['AccountID'] == account_id].iloc[0]
    industry = client_row['Business_Industry']
    subs = client_row['Subscribers']
    
    # Get Industry Data
    if industry in benchmark_df['Business_Industry'].values:
        industry_row = benchmark_df[benchmark_df['Business_Industry'] == industry].iloc[0]
    else:
        return None 
    
    metrics = {
        'Client_Name': str(client_row['AccountID']),
        'Industry': industry,
        'Subscribers': subs,
        'Customer_Since': client_row['Customer_Since'],
        'Benchmark_Results': {}
    }
    
    if subs > 0:
        # 1. Total Case Load Comparison
        client_total_rate = client_row['Total_Cases_Calculated'] / subs
        ind_total_rate = industry_row['Total_Cases_Rate']
        
        if ind_total_rate > 0:
            # Percentage difference calculation
            metrics['Case_Load_Diff'] = ((client_total_rate - ind_total_rate) / ind_total_rate) * 100
        else:
            metrics['Case_Load_Diff'] = 0
        
        # 2. Client Case Breakdown (Raw Numbers)
        metrics['Client_Case_Totals'] = {col: client_row[col] for col in CASE_COLUMNS}
        
        # 3. Utilization Comparison
        util_comparison = []
        for col in UTIL_COLUMNS:
            c_rate = client_row[col] / subs
            i_rate = industry_row[f"{col}_Rate"]
            
            diff = 0
            if i_rate > 0:
                diff = ((c_rate - i_rate) / i_rate) * 100
            
            util_comparison.append({
                'Metric': col.replace('_', ' '),
                'Client Rate': c_rate,
                'Industry Avg': i_rate,
                'Difference': diff # Will format as % later
            })
        metrics['Util_Comparison'] = pd.DataFrame(util_comparison)
        
    return metrics

def run_projection(subscribers, industry, benchmark_df):
    """Projects cases based on subscriber count and industry averages."""
    if benchmark_df.empty:
        return pd.DataFrame(), 0
        
    industry_row = benchmark_df[benchmark_df['Business_Industry'] == industry].iloc[0]
    
    projections = []
    total_proj = 0
    
    for col in CASE_COLUMNS:
        rate = industry_row[f"{col}_Rate"]
        proj_count = rate * subscribers
        total_proj += proj_count
        
        # Friendly Names
        display_name = col.replace('Medical_Cases_', 'Medical - ').replace('Security_Cases_', 'Security - ').replace('_', ' ')
        
        projections.append({
            'Case Type': display_name,
            'Projected Cases': proj_count
        })
        
    df = pd.DataFrame(projections)
    
    # Add Percentage Column
    if total_proj > 0:
        df['% of Total'] = (df['Projected Cases'] / total_proj) * 100
    else:
        df['% of Total'] = 0
        
    return df, total_proj


# --- Helper: Styling ---
def get_diff_color(val, invert=False):
    # Invert=True means Lower is Better (e.g. Cases)
    # Invert=False means Higher is Better (e.g. Utilization)
    if invert:
        if val < -10: return BENCHMARK_COLOR_GOOD
        if val > 10: return BENCHMARK_COLOR_BAD
    else:
        if val > 10: return BENCHMARK_COLOR_GOOD
        if val < -10: return BENCHMARK_COLOR_BAD
    return BRAND_COLOR_BLUE

# --- APP LAYOUT ---

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

# 1.1 Account Selector
ids = sorted(RAW_DATA_DF['AccountID'].unique())
# Default value is "Select here..."
selected_id_val = st.selectbox("1.1 Select Account ID", ["Select here..."] + ids)

if selected_id_val != "Select here...":
    # Only run logic if a valid ID is picked
    metrics = get_client_metrics(selected_id_val, RAW_DATA_DF, INDUSTRY_BENCHMARKS_DF)
    
    if metrics:
        # 1.2/1.3 Display Fields
        c1, c2, c3 = st.columns(3)
        c1.text_input("1.2 Business Industry", value=metrics['Industry'], disabled=True)
        c2.text_input("1.3 Number of Subscribers", value=f"{metrics['Subscribers']:,}", disabled=True)
        
        cust_since = metrics['Customer_Since']
        since_str = cust_since.strftime('%Y-%m-%d') if pd.notna(cust_since) else "N/A"
        c3.text_input("Customer Since", value=since_str, disabled=True)
        
        st.markdown('---')
        
        # Case Load Summary Box
        diff = metrics.get('Case_Load_Diff', 0)
        color = get_diff_color(diff, invert=True)
        
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
            # Case Breakdown Table
            breakdown_data = []
            if 'Client_Case_Totals' in metrics:
                for col, val in metrics['Client_Case_Totals'].items():
                    name = col.replace('Medical_Cases_', 'Med - ').replace('Security_Cases_', 'Sec - ').replace('_', ' ')
                    breakdown_data.append({'Case Type': name, 'Cases': val})
            
            bd_df = pd.DataFrame(breakdown_data)
            if not bd_df.empty:
                total_c = bd_df['Cases'].sum()
                bd_df['% of Total'] = (bd_df['Cases'] / total_c) * 100 if total_c > 0 else 0
                
                st.subheader("Client Case Breakdown")
                st.dataframe(
                    bd_df.style.format({'% of Total': '{:.1f}%'}), 
                    use_container_width=True, 
                    hide_index=True
                )

        st.write("")
        st.markdown(f'<h3 style="color:{BRAND_COLOR_DARK};">Utilization Analysis</h3>', unsafe_allow_html=True)
        
        # Utilization Table
        if 'Util_Comparison' in metrics:
            util_df = metrics['Util_Comparison']
            
            def style_util(val):
                color = get_diff_color(val, invert=False)
                return f'color: {color}; font-weight: bold'

            # Display with % formatting
            st.dataframe(
                util_df.style.format({
                    'Client Rate': '{:.4f}',
                    'Industry Avg': '{:.4f}',
                    'Difference': '{:+.1f}%' # Added % symbol here
                }).applymap(style_util, subset=['Difference']),
                use_container_width=True,
                hide_index=True
            )

else:
    # Reset metrics so section 2 defaults to standard values if no account selected
    metrics = None

st.markdown('---')


# --- SECTION 2: Projection ---
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">2. Case Projection Model</h2>', unsafe_allow_html=True)

p1, p2 = st.columns(2)
with p1:
    if not INDUSTRY_BENCHMARKS_DF.empty:
        industries = sorted(INDUSTRY_BENCHMARKS_DF['Business_Industry'].unique())
        # Default: Match selected account if possible
        def_ix = 0
        if metrics and metrics['Industry'] in industries:
             def_ix = industries.index(metrics['Industry'])
        
        proj_ind = st.selectbox("2.1 Select Industry", industries, index=def_ix)
    else:
        proj_ind = None
        st.warning("No industry data available.")

with p2:
    # Default: Match selected account subscribers if available, else 0
    def_sub = int(metrics['Subscribers']) if metrics else 0
    # Start at 0 as requested
    proj_sub = st.number_input("2.2 Subscribers", min_value=0, value=def_sub)

if proj_ind and proj_sub > 0 and not INDUSTRY_BENCHMARKS_DF.empty:
    proj_df, total_p = run_projection(proj_sub, proj_ind, INDUSTRY_BENCHMARKS_DF)
    
    c_p_metric, c_p_table = st.columns([1, 2])
    with c_p_metric:
        st.metric("Projected Total Cases (Annual)", f"{total_p:,.1f}")
    
    with c_p_table:
        st.dataframe(
            proj_df.style.format({
                'Projected Cases': '{:.1f}', 
                '% of Total': '{:.1f}%' # Added % symbol here
            }),
            use_container_width=True,
            hide_index=True
        )

# --- Footer ---
st.markdown('---')
st.markdown(f"""
<div style="text-align:center; color:gray; padding:20px;">
    <a href="https://www.internationalsos.com/get-in-touch?utm_source=benchmarkingreport" target="_blank">
        <button style="background-color:{BRAND_COLOR_ORANGE}; color:white; border:none; padding:10px 20px; border-radius:5px; cursor:pointer;">
            Get in Touch
        </button>
    </a>
</div>
""", unsafe_allow_html=True)
