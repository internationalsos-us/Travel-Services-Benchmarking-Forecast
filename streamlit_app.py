import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
from datetime import datetime, date

# --- Global Configuration ---
BRAND_COLOR_BLUE = "#2f4696"
BRAND_COLOR_DARK = "#232762"
BENCHMARK_COLOR_GOOD = "#009354"
BENCHMARK_COLOR_BAD = "#D4002C"
BRAND_COLOR_ORANGE = "#EF820F"

# --- Column Mapping (Based on your specific instructions A-S) ---
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
        
        # Rename logic handling potential missing underscores or spaces in raw file
        rename_dict = {}
        file_cols = df.columns.tolist()
        
        for target_col, internal_name in COLUMN_MAP.items():
            # Try exact match
            if target_col in file_cols:
                rename_dict[target_col] = internal_name
            else:
                # Fallback: Try fuzzy match (ignore case, spaces, underscores, &)
                normalized_target = target_col.replace('&', '').replace(' ', '').replace('_', '').lower()
                found = False
                for file_col in file_cols:
                    normalized_file = file_col.replace('&', '').replace(' ', '').replace('_', '').lower()
                    if normalized_file == normalized_target:
                        rename_dict[file_col] = internal_name
                        found = True
                        break
                if not found:
                    # Critical column missing? Only warn if it's a core mapping failure
                    pass 

        df = df.rename(columns=rename_dict)
        
        # Verify we have all case columns mapped
        existing_case_cols = [c for c in CASE_COLUMNS if c in df.columns]
        if len(existing_case_cols) < len(CASE_COLUMNS):
             st.warning(f"Note: Some case columns could not be mapped from the CSV. Found {len(existing_case_cols)} of {len(CASE_COLUMNS)}.")

        # 2. Data Type Conversion
        if 'AccountID' in df.columns:
            df['AccountID'] = df['AccountID'].astype(str)
        if 'Customer_Since' in df.columns:
            df['Customer_Since'] = pd.to_datetime(df['Customer_Since'], errors='coerce')
        
        # 3. Calculate Total Cases (Sum of all case columns present)
        # Fill NaNs with 0 before summing
        cols_to_sum = [c for c in CASE_COLUMNS if c in df.columns]
        df[cols_to_sum] = df[cols_to_sum].fillna(0)
        df['Total_Cases_Calculated'] = df[cols_to_sum].sum(axis=1)
        
        # Pre-calculate rates for charts
        if 'Subscribers' in df.columns:
             # Avoid division by zero
             df['Cases_Per_Subscriber'] = df.apply(lambda x: x['Total_Cases_Calculated'] / x['Subscribers'] if x['Subscribers'] > 0 else 0, axis=1)
             
             cols_util_sum = [c for c in UTIL_COLUMNS if c in df.columns]
             df['Total_Utilization_Count'] = df[cols_util_sum].sum(axis=1)
             df['Utilization_Per_Subscriber'] = df.apply(lambda x: x['Total_Utilization_Count'] / x['Subscribers'] if x['Subscribers'] > 0 else 0, axis=1)

        return df
    except FileNotFoundError:
        st.error("Data file 'app_raw_data.csv' not found. Please upload it.")
        return pd.DataFrame()

RAW_DATA_DF = load_data()


# --- Core Logic Functions ---

@st.cache_data
def calculate_industry_averages(df):
    """Calculates weighted average rates per subscriber for each industry."""
    if df.empty or 'Business_Industry' not in df.columns:
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
        
        # Case Rates
        existing_case_cols = [c for c in CASE_COLUMNS if c in df.columns]
        for col in existing_case_cols:
            total_cases_in_industry = group[col].sum()
            industry_data[f"{col}_Rate"] = total_cases_in_industry / total_subscribers
            
        # Utilization Rates
        existing_util_cols = [c for c in UTIL_COLUMNS if c in df.columns]
        for col in existing_util_cols:
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
        
    client_row = raw_df[raw_df['AccountID'] == account_id].iloc[0]
    client_industry = client_row['Business_Industry']
    client_subs = client_row['Subscribers']
    
    if client_industry in benchmark_df['Business_Industry'].values:
        industry_row = benchmark_df[benchmark_df['Business_Industry'] == client_industry].iloc[0]
    else:
        return None 
    
    metrics = {
        'Client_Name': str(client_row['AccountID']),
        'Industry': client_industry,
        'Subscribers': client_subs,
        'Customer_Since': client_row['Customer_Since'],
        'Benchmark_Results': {},
        'Industry_Case_Rates': {} # Store for chart
    }
    
    # Populate industry rates for chart
    existing_case_cols = [c for c in CASE_COLUMNS if c in raw_df.columns]
    for col in existing_case_cols:
         metrics['Industry_Case_Rates'][col] = industry_row[f"{col}_Rate"]
    
    if client_subs > 0:
        # 1. Case Load Comparison
        client_total_rate = client_row['Total_Cases_Calculated'] / client_subs
        ind_total_rate = industry_row['Total_Cases_Rate']
        
        if ind_total_rate > 0:
            metrics['Case_Load_Diff'] = ((client_total_rate - ind_total_rate) / ind_total_rate) * 100
        else:
            metrics['Case_Load_Diff'] = 0
        
        # 2. Client Case Breakdown
        metrics['Client_Case_Totals'] = {col: client_row[col] for col in existing_case_cols}
        
        # 3. Utilization Comparison
        util_comparison = []
        existing_util_cols = [c for c in UTIL_COLUMNS if c in raw_df.columns]
        for col in existing_util_cols:
            c_rate = client_row[col] / client_subs
            i_rate = industry_row[f"{col}_Rate"]
            
            diff = 0
            if i_rate > 0:
                diff = ((c_rate - i_rate) / i_rate) * 100
            
            util_comparison.append({
                'Metric': col.replace('_', ' '),
                'Client Rate': c_rate,
                'Industry Avg': i_rate,
                'Difference': diff 
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
    
    # Only project for columns that exist in the benchmark DF (which means they were in raw DF)
    valid_case_cols = [c for c in CASE_COLUMNS if f"{c}_Rate" in benchmark_df.columns]
    
    for col in valid_case_cols:
        rate = industry_row[f"{col}_Rate"]
        proj_count = rate * subscribers
        total_proj += proj_count
        
        display_name = col.replace('Medical_Cases_', 'Med - ').replace('Security_Cases_', 'Security - ').replace('_', ' ')
        
        projections.append({
            'Case Type': display_name,
            'Projected Cases': proj_count
        })
        
    df = pd.DataFrame(projections)
    
    if total_proj > 0:
        df['% of Total'] = (df['Projected Cases'] / total_proj) 
    else:
        df['% of Total'] = 0
        
    return df, total_proj


# --- Helper: Styling ---
def get_diff_color(val, invert=False):
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
st.write("Compare client assistance activity against industry peers.")
st.markdown('---')

# Section 1
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">1. Account Benchmarking & Utilization Analysis</h2>', unsafe_allow_html=True)

if RAW_DATA_DF.empty: st.stop()

ids = sorted(RAW_DATA_DF['AccountID'].unique())
sel_id = st.selectbox("1.1 Select Account ID", ["Select here..."] + ids)

metrics = None
if sel_id != "Select here...":
    metrics = get_client_metrics(sel_id, RAW_DATA_DF, INDUSTRY_BENCHMARKS_DF)
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.text_input("1.2 Business Industry", value=metrics['Industry'], disabled=True)
        c2.text_input("1.3 Number of Subscribers", value=f"{metrics['Subscribers']:,}", disabled=True)
        c3.text_input("Customer Since", value=metrics['Customer_Since'].strftime('%Y-%m-%d') if pd.notna(metrics['Customer_Since']) else "N/A", disabled=True)
        
        st.markdown('---')
        
        diff = metrics.get('Case_Load_Diff', 0)
        col = get_diff_color(diff, invert=True)
        
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
                <h3 style="margin:0; color:{BRAND_COLOR_DARK}">Total Case Load</h3>
                <h1 style="font-size:48px; margin:10px 0; color:{col}">{abs(diff):.1f}%</h1>
                <p style="font-size:18px; font-weight:bold; color:{col}">
                    {'BELOW' if diff < 0 else 'ABOVE'} Industry Avg
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with cc2:
            bd_data = []
            if 'Client_Case_Totals' in metrics:
                for k, v in metrics['Client_Case_Totals'].items():
                    bd_data.append({'Case Type': k.replace('Medical_Cases_', 'Med - ').replace('Security_Cases_', 'Sec - ').replace('_', ' '), 'Cases': v})
            bd_df = pd.DataFrame(bd_data)
            # Filter out rows with 0 cases
            bd_df = bd_df[bd_df['Cases'] > 0]
            
            if not bd_df.empty:
                tot = bd_df['Cases'].sum()
                bd_df['% of Total'] = (bd_df['Cases'] / tot)
                
                st.subheader("Client Case Breakdown")
                st.dataframe(
                    bd_df,
                    column_config={
                        "% of Total": st.column_config.NumberColumn(
                            "% of Total",
                            format="%.1f%"
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No cases recorded.")
        
        # --- NEW CHART: Case Rate Comparison (Bar with Line) ---
        st.write("")
        st.subheader("Client vs. Industry: Case Rate Comparison")
        
        chart_data = []
        subs = metrics['Subscribers']
        
        if subs > 0:
            for k, v in metrics['Client_Case_Totals'].items():
                friendly_name = k.replace('Medical_Cases_', 'Med-').replace('Security_Cases_', 'Sec-').replace('_', ' ')
                
                client_rate_1k = (v / subs) * 1000
                # Ensure we get the corresponding industry rate
                ind_rate_1k = metrics['Industry_Case_Rates'].get(k, 0) * 1000
                
                chart_data.append({
                    'Case Type': friendly_name,
                    'Client Rate (per 1k)': client_rate_1k,
                    'Industry Rate (per 1k)': ind_rate_1k
                })
            
            chart_df = pd.DataFrame(chart_data)
            # Filter for cleaner chart: show if either rate is significant
            chart_df = chart_df[(chart_df['Client Rate (per 1k)'] > 0.01) | (chart_df['Industry Rate (per 1k)'] > 0.01)]

            if not chart_df.empty:
                fig_compare = go.Figure()

                # Bar for Client
                fig_compare.add_trace(go.Bar(
                    x=chart_df['Case Type'],
                    y=chart_df['Client Rate (per 1k)'],
                    name='Client Rate',
                    marker_color=BRAND_COLOR_BLUE
                ))

                # Line for Industry Average (Red Dashed)
                fig_compare.add_trace(go.Scatter(
                    x=chart_df['Case Type'],
                    y=chart_df['Industry Rate (per 1k)'],
                    name='Industry Avg',
                    mode='lines+markers',
                    line=dict(color='red', width=2, dash='dash'), 
                    marker=dict(symbol='circle', size=8, color='red')
                ))

                fig_compare.update_layout(
                    yaxis_title="Rate per 1k Subscribers",
                    xaxis_title="",
                    legend_title="",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
            else:
                st.info("No significant case data to display in chart.")


        st.write("")
        st.markdown(f'<h3 style="color:{BRAND_COLOR_DARK};">Utilization Analysis</h3>', unsafe_allow_html=True)
        if 'Util_Comparison' in metrics:
            udf = metrics['Util_Comparison']
            st.dataframe(
                udf,
                column_config={
                    "Difference": st.column_config.NumberColumn(
                        "Difference",
                        format="%+.1f%" # Adds + sign and % symbol
                    )
                },
                use_container_width=True, 
                hide_index=True
            )

st.markdown('---')

# --- SECTION 2: CORRELATION GRAPH ---
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">2. Correlation: Utilization vs. Case Load</h2>', unsafe_allow_html=True)
st.write("This chart compares **Total Utilization** against **Total Case Load**. Outliers (top 5%) are removed for clarity.")

industries_list = sorted(RAW_DATA_DF['Business_Industry'].dropna().unique().tolist())
selected_industry_filter = st.selectbox("Filter Chart by Industry", ["All Industries"] + industries_list)

plot_df = RAW_DATA_DF[RAW_DATA_DF['Subscribers'] > 0].copy()

if selected_industry_filter != "All Industries":
    plot_df = plot_df[plot_df['Business_Industry'] == selected_industry_filter]

if not plot_df.empty:
    util_cap = plot_df['Utilization_Per_Subscriber'].quantile(0.95)
    plot_df_filtered = plot_df[plot_df['Utilization_Per_Subscriber'] <= util_cap].copy()
else:
    plot_df_filtered = pd.DataFrame()

if sel_id != "Select here...":
    selected_row = RAW_DATA_DF[RAW_DATA_DF['AccountID'] == str(sel_id)]
    if not selected_row.empty:
         if str(sel_id) not in plot_df_filtered['AccountID'].values:
             plot_df_filtered = pd.concat([plot_df_filtered, selected_row])

    plot_df_filtered['Client_Type'] = plot_df_filtered['AccountID'].apply(lambda x: 'Selected Client' if str(x) == str(sel_id) else 'Other Clients')
    color_map = {'Selected Client': BRAND_COLOR_ORANGE, 'Other Clients': BRAND_COLOR_BLUE}
else:
    plot_df_filtered['Client_Type'] = 'All Clients'
    color_map = {'All Clients': BRAND_COLOR_BLUE}

if not plot_df_filtered.empty:
    fig = px.scatter(
        plot_df_filtered,
        x="Utilization_Per_Subscriber",
        y="Cases_Per_Subscriber",
        color="Client_Type",
        color_discrete_map=color_map,
        hover_data=["AccountID", "Business_Industry", "Subscribers"],
        title=f"Utilization vs. Case Rate ({selected_industry_filter})",
        labels={
            "Utilization_Per_Subscriber": "Total Utilization Actions per Subscriber",
            "Cases_Per_Subscriber": "Total Cases per Subscriber"
        },
        height=500
    )
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for this selection.")

st.markdown('---')

# --- SECTION 3: Projection ---
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">3. Case Projection Model</h2>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    if not INDUSTRY_BENCHMARKS_DF.empty:
        inds = sorted(INDUSTRY_BENCHMARKS_DF['Business_Industry'].unique())
        d_ix = inds.index(metrics['Industry']) if metrics and metrics['Industry'] in inds else 0
        p_ind = st.selectbox("3.1 Select Industry", inds, index=d_ix)
    else:
        p_ind = None
with c2:
    d_sub = int(metrics['Subscribers']) if metrics else 0
    p_sub = st.number_input("3.2 Subscribers", min_value=0, value=d_sub)

if p_ind and p_sub > 0 and not INDUSTRY_BENCHMARKS_DF.empty:
    pdf, tot_p = run_projection(p_sub, p_ind, INDUSTRY_BENCHMARKS_DF)
    m_col, t_col = st.columns([1, 2])
    with m_col: st.metric("Projected Annual Cases", f"{tot_p:,.1f}")
    with t_col: 
        pdf = pdf[pdf['Projected Cases'] > 0.01]
        st.dataframe(
            pdf,
            column_config={
                "% of Total": st.column_config.NumberColumn(
                    "% of Total",
                    format="%.1f%"
                ),
                "Projected Cases": st.column_config.NumberColumn(
                    "Projected Cases",
                    format="%.1f"
                )
            },
            use_container_width=True,
            hide_index=True
        )

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
