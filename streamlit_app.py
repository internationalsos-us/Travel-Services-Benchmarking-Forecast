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
# --- Hardcoded Industry Risk Profiles for Section 2 ---
INDUSTRY_RISK_PROFILES = {
    "Technology": {
        "summary": "High volume, low-risk travel. Focus is typically on general medical assistance, minimizing travel disruption, and ensuring high utilization of digital resources like the App/Portal for pre-trip planning.",
        "icon": "📱"
    },
    "Energy & Mining": {
        "summary": "High-risk, remote deployments. Case severity is often high, with a focus on Medical Evacuation (high cost/low frequency) and robust security services (e.g., Security Intervention and Active Monitoring).",
        "icon": "⛏️"
    },
    "Financial Services": {
        "summary": "High-frequency business travel in established locations. Key risks involve security concerns in politically unstable regions and high-value personnel needing rapid, discreet support.",
        "icon": "🏦"
    },
    "Manufacturing": {
        "summary": "Mixed risk profile, with global supply chains. Cases often center on general travel illness and low-level security incidents (e.g., Referrals and Information & Analysis).",
        "icon": "⚙️"
    },
    "Other Industries": {
        "summary": "This industry shows a varied risk landscape, emphasizing general travel assistance and maintaining situational awareness through Alerts and Pre-Trip Advisories. Focus on comprehensive, generalized support.",
        "icon": "🌍"
    }
}
# --- Column Mapping ---
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
CASE_COLUMNS = [
    "Travel_Cases", "Medical_Cases_IA", "Medical_Cases_OutPatient",
    "Medical_Cases_InPatient", "Medical_Cases_Evac", "Security_Cases_IA",
    "Security_Cases_Referrals", "Security_Cases_Intervention",
    "Security_Cases_Evac", "Security_Cases_ActiveMonitoring"
]
UTIL_COLUMNS = [
    "App_and_Portal_Sessions", "Alerts_Sent_to_Travelers",
    "Pre_Trip_Advisories_Sent", "E_Learning_Completed_Courses"
]
# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        # NOTE: Assumes 'app_raw_data.csv' is available in the environment
        df = pd.read_csv("app_raw_data.csv") 
        df.columns = df.columns.str.strip()
        
        rename_dict = {}
        for req_col, internal in COLUMN_MAP.items():
            if req_col in df.columns:
                rename_dict[req_col] = internal
            else:
                # Fallback fuzzy match
                for col in df.columns:
                    # Simplified fuzzy match for robustness
                    if col.lower().replace(' ','').replace('_','') == req_col.lower().replace(' ','').replace('_',''):
                        rename_dict[col] = internal
                        break
        
        df = df.rename(columns=rename_dict)
        df['AccountID'] = df['AccountID'].astype(str)
        df['Customer_Since'] = pd.to_datetime(df['Customer_Since'], errors='coerce')
        
        # Ensure columns exist and fill NaN with 0
        for col in CASE_COLUMNS + UTIL_COLUMNS + ['Subscribers']:
             if col not in df.columns:
                 df[col] = 0
             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['Total_Cases_Calculated'] = df[CASE_COLUMNS].sum(axis=1)
        
        # Pre-calculate rates
        if 'Subscribers' in df.columns:
             df['Cases_Per_Subscriber'] = df.apply(lambda x: x['Total_Cases_Calculated'] / x['Subscribers'] if x['Subscribers'] > 0 else 0, axis=1)
             cols_util_sum = [c for c in UTIL_COLUMNS if c in df.columns]
             df['Total_Utilization_Count'] = df[cols_util_sum].sum(axis=1)
             df['Utilization_Per_Subscriber'] = df.apply(lambda x: x['Total_Utilization_Count'] / x['Subscribers'] if x['Subscribers'] > 0 else 0, axis=1)
        return df
    except FileNotFoundError:
        st.error("Data file 'app_raw_data.csv' not found. Please ensure it is uploaded.")
        return pd.DataFrame()
RAW_DATA_DF = load_data()
# --- Logic Functions ---
@st.cache_data
def calculate_industry_averages(df):
    if df.empty: return pd.DataFrame()
    industry_groups = df.groupby('Business_Industry')
    benchmarks = []
    for industry, group in industry_groups:
        total_sub = group['Subscribers'].sum()
        if total_sub == 0: continue
        
        data = {'Business_Industry': industry, 'Total_Subscribers_Base': total_sub}
        for col in CASE_COLUMNS + UTIL_COLUMNS:
            data[f"{col}_Rate"] = group[col].sum() / total_sub
        
        data['Total_Cases_Rate'] = group['Total_Cases_Calculated'].sum() / total_sub
        
        # Add industry average of Utilization_Per_Subscriber for downstream use
        util_rate_col = [c for c in group.columns if c == 'Utilization_Per_Subscriber']
        if util_rate_col:
            data['Utilization_Per_Subscriber'] = group['Total_Utilization_Count'].sum() / total_sub
        
        benchmarks.append(data)
    return pd.DataFrame(benchmarks)
INDUSTRY_BENCHMARKS_DF = calculate_industry_averages(RAW_DATA_DF)
def get_client_metrics(account_id, raw_df, benchmark_df):
    if raw_df.empty: return None
    client_row = raw_df[raw_df['AccountID'] == account_id].iloc[0]
    industry = client_row['Business_Industry']
    subs = client_row['Subscribers']
    
    if industry in benchmark_df['Business_Industry'].values:
        ind_row = benchmark_df[benchmark_df['Business_Industry'] == industry].iloc[0]
    else:
        # Fallback to general industry data if client industry is not in benchmarks
        return None
        
    metrics = {
        'Industry': industry, 'Subscribers': subs, 
        'Customer_Since': client_row['Customer_Since'],
        'Client_Case_Totals': {col: client_row[col] for col in CASE_COLUMNS},
        'Industry_Case_Rates': {col: ind_row[f"{col}_Rate"] for col in CASE_COLUMNS}
    }
    
    util_data = [] 
    
    if subs > 0:
        c_rate = client_row['Total_Cases_Calculated'] / subs
        i_rate = ind_row['Total_Cases_Rate']
        metrics['Case_Load_Diff'] = ((c_rate - i_rate)/i_rate)*100 if i_rate > 0 else 0
        
        util_comp = []
        for col in UTIL_COLUMNS:
            u_rate = client_row[col] / subs
            ui_rate = ind_row[f"{col}_Rate"]
            diff = ((u_rate - ui_rate)/ui_rate)*100 if ui_rate > 0 else 0
            
            util_comp.append({
                'Metric': col.replace('_',' '), 'Client Rate': u_rate, 
                'Industry Avg': ui_rate, 'Difference': diff 
            })
            # Store raw data for any future correlation needs
            util_data.append({
                'Metric': col, 
                'Client_Rate': u_rate, 
                'Industry_Rate': ui_rate
            })
            
        metrics['Util_Comparison'] = pd.DataFrame(util_comp)
        metrics['Raw_Util_Rates'] = pd.DataFrame(util_data)
    return metrics
def run_projection(subs, industry, bench_df):
    if bench_df.empty: return pd.DataFrame(), 0
    row = bench_df[bench_df['Business_Industry'] == industry].iloc[0]
    projs = []
    total = 0
    for col in CASE_COLUMNS:
        val = row[f"{col}_Rate"] * subs
        total += val
        projs.append({
            'Case Type': col.replace('Medical_Cases_', 'Med - ').replace('Security_Cases_', 'Sec - ').replace('_', ' '),
            'Projected Cases': val
        })
    df = pd.DataFrame(projs)
    df['% of Total'] = (df['Projected Cases']/total * 100) if total > 0 else 0
    return df, total

@st.cache_data
def get_impact_factor(df_industry):
    """
    Performs linear regression (Cases/Sub ~ Utilization/Sub) and returns the slope (m) and intercept (c).
    This function is kept but no longer used in the main flow.
    """
    if df_industry.empty or len(df_industry) < 2:
        return 0.0, 0.0 # slope, intercept
    
    # Filter out outliers (same as in Section 2) for a more robust fit
    util_cap = df_industry['Utilization_Per_Subscriber'].quantile(0.95)
    df_filtered = df_industry[df_industry['Utilization_Per_Subscriber'] <= util_cap].copy()
    
    X = df_filtered['Utilization_Per_Subscriber']
    Y = df_filtered['Cases_Per_Subscriber']
    
    if len(X) < 2:
         return 0.0, 0.0
         
    # Perform linear regression: Y = mX + c
    m, c = np.polyfit(X, Y, 1)
    return m, c

def get_diff_color(val, invert=False):
    # val is percentage (e.g. 10.0 for 10%)
    if invert: return BENCHMARK_COLOR_GOOD if val < -10 else (BENCHMARK_COLOR_BAD if val > 10 else BRAND_COLOR_BLUE)
    return BENCHMARK_COLOR_GOOD if val > 10 else (BENCHMARK_COLOR_BAD if val < -10 else BRAND_COLOR_BLUE)
# --- APP LAYOUT ---
st.set_page_config(page_title="International SOS | Benchmarking", layout="wide")
st.markdown(f"""
<div style="background-color:{BRAND_COLOR_DARK}; padding:20px; display:flex; align-items:center;">
    <img src="https://images.learn.internationalsos.com/EloquaImages/clients/InternationalSOS/%7B0769a7db-dae2-4ced-add6-d1a73cb775d5%7D_International_SOS_white_hr_%281%29.png" style="height:50px; margin-right:20px;">
    <h1 style="color:white; margin:0; font-size:24px;">Travel Services Benchmarking and Forecast</h1>
</div>
""", unsafe_allow_html=True)
st.write("")
# 1. Title Change
st.markdown(f'<h1 style="color:{BRAND_COLOR_DARK};">Assistance Services Benchmarking Report</h1>', unsafe_allow_html=True)
st.write("Compare client assistance activity against industry peers.")
st.markdown('---')
# Section 1
# 2. Section 1 Title Change + Subtitle
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">1. Existing Client Utilization and Benchmarking Analysis</h2>', unsafe_allow_html=True)
st.markdown("**(Actual activity for a 12-month period)**") # Subtitle added here
if RAW_DATA_DF.empty: st.stop()
# Ensure IDs are sorted for easy typing/searching
ids = sorted(RAW_DATA_DF['AccountID'].unique())
# Initialize sel_id, metrics globally for downstream use
sel_id = "Select here..."
metrics = None
if 'sel_id_input' not in st.session_state:
    st.session_state.sel_id_input = "Select here..."
    
# Updated to use st.selectbox, which supports typing to filter/search
sel_id = st.selectbox(
    "1.1 Select Account ID (Start typing to search)", 
    ["Select here..."] + ids, 
    key='sel_id_input'
)

if sel_id != "Select here...":
    metrics = get_client_metrics(sel_id, RAW_DATA_DF, INDUSTRY_BENCHMARKS_DF)
    if metrics:
        c1, c2, c3 = st.columns(3)
        st.session_state.current_industry = metrics['Industry'] 
        st.session_state.current_subs = metrics['Subscribers'] 
        
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
            total_cases = 0
            if 'Client_Case_Totals' in metrics:
                for k, v in metrics['Client_Case_Totals'].items():
                    # Populate data for the breakdown table
                    bd_data.append({'Case Type': k.replace('Medical_Cases_', 'Med - ').replace('Security_Cases_', 'Sec - ').replace('_', ' '), 'Cases': v})
                    total_cases += v
            
            bd_df = pd.DataFrame(bd_data)
            
            # Filter to show only cases > 0
            bd_df_filtered = bd_df[bd_df['Cases'] > 0].copy() 
            
            if not bd_df_filtered.empty or total_cases > 0:
                tot = total_cases
                
                # Format % of Total based on the grand total
                if tot > 0:
                    bd_df_filtered['% of Total'] = (bd_df_filtered['Cases'] / tot * 100).map('{:.1f}%'.format)
                else:
                    bd_df_filtered['% of Total'] = '0.0%'
                    
                # Add Total Row
                total_row = pd.DataFrame([{
                    'Case Type': 'Total Cases', 
                    'Cases': tot, 
                    '% of Total': '100.0%' if tot > 0 else '0.0%'
                }])
                
                # Concatenate the total row to the filtered breakdown data
                bd_df_final = pd.concat([bd_df_filtered, total_row], ignore_index=True)
                
                # Styling function to bold the last row (Total Cases)
                def style_total_row(row):
                    is_total = row['Case Type'] == 'Total Cases'
                    # Apply a lighter background to the total row
                    styles = ['font-weight: bold; background-color: #e6e6e6;'] * len(row) 
                    return styles if is_total else [''] * len(row)
                
                # 3. Client Case Breakdown Title Change
                st.subheader("Actual Client Cases by Type")
                st.dataframe(
                    bd_df_final.style.apply(style_total_row, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No cases recorded.")
        
        # --- CHART: Case Rate Comparison ---
        st.write("")
        # 4. Case Rate Comparison Title Change
        st.subheader("Case Rate Frequency compared to Industry Average")
        
        chart_data = []
        subs = metrics['Subscribers']
        if subs > 0:
            for k, v in metrics['Client_Case_Totals'].items():
                friendly_name = k.replace('Medical_Cases_', 'Med-').replace('Security_Cases_', 'Sec-').replace('_', ' ')
                client_rate_1k = (v / subs) * 1000
                ind_rate_1k = metrics['Industry_Case_Rates'].get(k, 0) * 1000
                chart_data.append({'Case Type': friendly_name, 'Client Rate (per 1k)': client_rate_1k, 'Industry Rate (per 1k)': ind_rate_1k})
            
            chart_df = pd.DataFrame(chart_data)
            chart_df = chart_df[(chart_df['Client Rate (per 1k)'] > 0.01) | (chart_df['Industry Rate (per 1k)'] > 0.01)]
            if not chart_df.empty:
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(x=chart_df['Case Type'], y=chart_df['Client Rate (per 1k)'], name='Client Rate', marker_color=BRAND_COLOR_BLUE))
                
                # Industry Average Line (Red Dashed)
                fig_compare.add_trace(go.Scatter(
                    x=chart_df['Case Type'], 
                    y=chart_df['Industry Rate (per 1k)'],
                    name='Industry Avg',
                    mode='lines+markers',
                    line=dict(color='red', width=2, dash='dash'), 
                    marker=dict(symbol='circle', size=8, color='red')
                ))
                
                fig_compare.update_layout(yaxis_title="Rate per 1k Subscribers", xaxis_title="", legend_title="", barmode='group', height=450)
                st.plotly_chart(fig_compare, use_container_width=True)
            else:
                st.info("No significant data.")
        st.write("")
        # 5. Utilization Analysis Title Change
        st.markdown(f'<h3 style="color:{BRAND_COLOR_DARK};">Digital Utilization by Type</h3>', unsafe_allow_html=True)
        if 'Util_Comparison' in metrics:
            udf = metrics['Util_Comparison']
            
            # Pre-format columns to strings to avoid Streamlit formatting errors
            udf['Client Rate'] = udf['Client Rate'].map('{:.4f}'.format)
            udf['Industry Avg'] = udf['Industry Avg'].map('{:.4f}'.format)
            
            # Store raw difference for color logic before formatting
            diff_values = udf['Difference'].copy()
            udf['Difference'] = udf['Difference'].map('{:+.1f}%'.format)
            
            def style_util(row):
                # Access original numeric value via index if possible, or re-parse
                val = diff_values[row.name]
                color = get_diff_color(val, invert=False)
                return [f'color: {color}; font-weight: bold' if col == 'Difference' else '' for col in row.index]
            st.dataframe(
                udf.style.apply(style_util, axis=1),
                use_container_width=True, 
                hide_index=True
            )
st.markdown('---')
# --- SECTION 2: CORRELATION GRAPH ---
# 6. Correlation Section Title Change
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">2. Digital and Case Utilization Benchmarking</h2>', unsafe_allow_html=True)
# 7. Correlation Subtext Change
st.write("Compare digital utilization (alerts, app + portal use, eLearning, etc) against Assistance case utilization, by subscriber population. Outliers (top 5%) have been removed")
inds = sorted(RAW_DATA_DF['Business_Industry'].dropna().unique().tolist())
# Initialize sel_ind globally
sel_ind = "All Industries"
if 'sel_ind_chart' not in st.session_state:
    st.session_state.sel_ind_chart = "All Industries"
    
sel_ind = st.selectbox("Filter Chart by Industry", ["All Industries"] + inds, key='sel_ind_chart')

plot_df = RAW_DATA_DF[RAW_DATA_DF['Subscribers'] > 0].copy()
if sel_ind != "All Industries":
    plot_df = plot_df[plot_df['Business_Industry'] == sel_ind]

# Calculate filtered data (excluding top 5% of utilization for visualization consistency)
if not plot_df.empty:
    util_cap = plot_df['Utilization_Per_Subscriber'].quantile(0.95)
    plot_df_filtered = plot_df[plot_df['Utilization_Per_Subscriber'] <= util_cap].copy()
else:
    plot_df_filtered = pd.DataFrame()
    
# Highlight selected client if applicable
if sel_id != "Select here...":
    sel_row = RAW_DATA_DF[RAW_DATA_DF['AccountID'] == str(sel_id)]
    # Only concatenate if the client is not already in the filtered set (i.e., they are an outlier)
    if not sel_row.empty and str(sel_id) not in plot_df_filtered['AccountID'].values:
             plot_df_filtered = pd.concat([plot_df_filtered, sel_row])
             
    plot_df_filtered['Client_Type'] = plot_df_filtered['AccountID'].apply(lambda x: 'Selected Client' if str(x) == str(sel_id) else 'Other Clients')
    cmap = {'Selected Client': BRAND_COLOR_ORANGE, 'Other Clients': BRAND_COLOR_BLUE}
else:
    plot_df_filtered['Client_Type'] = 'All Clients'
    cmap = {'All Clients': BRAND_COLOR_BLUE}

# 8. X-Axis Label Change & 9. Y-Axis Label Change
X_AXIS_LABEL = "Digital Utilization/Subscriber"
Y_AXIS_LABEL = "Case Utilization/Subscriber"

if not plot_df_filtered.empty:
    # Calculate trendline data and draw it if a specific industry is selected
    if len(plot_df) >= 2 and sel_ind != "All Industries":
        # Calculate slope and intercept for the selected industry (using the original, non-highlighted data)
        trend_m, trend_c = get_impact_factor(plot_df)
        
        # Create a trend line
        x_min = plot_df['Utilization_Per_Subscriber'].min()
        x_max = plot_df['Utilization_Per_Subscriber'].max()
        x_range = np.linspace(x_min, x_max, 100)
        y_trend = trend_m * x_range + trend_c
        
        # Ensure trendline points are positive for plotting context
        y_trend = np.maximum(y_trend, 0)

        # Create the initial scatter plot
        fig = px.scatter(
            plot_df_filtered, x="Utilization_Per_Subscriber", y="Cases_Per_Subscriber",
            color="Client_Type", color_discrete_map=cmap,
            hover_data=["AccountID", "Business_Industry", "Subscribers"],
            labels={"Utilization_Per_Subscriber": X_AXIS_LABEL, "Cases_Per_Subscriber": Y_AXIS_LABEL},
            height=500
        )
        
        # Add trendline to the scatter plot
        fig.add_trace(go.Scatter(x=x_range, y=y_trend, mode='lines', 
                                 name=f'Trend Line (m={trend_m:+.4f})', 
                                 line=dict(color=BRAND_COLOR_DARK, width=2, dash='dot')))
                              
    else:
         fig = px.scatter(
            plot_df_filtered, x="Utilization_Per_Subscriber", y="Cases_Per_Subscriber",
            color="Client_Type", color_discrete_map=cmap,
            hover_data=["AccountID", "Business_Industry", "Subscribers"],
            labels={"Utilization_Per_Subscriber": X_AXIS_LABEL, "Cases_Per_Subscriber": Y_AXIS_LABEL},
            height=500
        )
        
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available.")

# --- Dynamic Industry Profile (New Element in Section 2) ---
if sel_ind != "All Industries":
    
    # 1. Determine which profile to use (use fallback if specific profile is missing)
    profile_key = sel_ind if sel_ind in INDUSTRY_RISK_PROFILES else "Other Industries"
    profile = INDUSTRY_RISK_PROFILES[profile_key]
    
    st.markdown(f'<h3 style="color:{BRAND_COLOR_DARK}; margin-top: 30px;">{profile["icon"]} {sel_ind} Industry Risk Profile</h3>', unsafe_allow_html=True)
    
    # 2. Summary Box
    st.info(profile["summary"])

    # 10. Remove entirely the section that starts with "Top 3 Case rates in this industry" (Removed logic here)
    # The 'Top 3 Case Types' section and its related logic/display were removed below this comment.

st.markdown('---')
# --- SECTION 3: Projection ---
# 11. Section 3 Title Change
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">3. New Client or Annual Cases Forecast</h2>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    # Default to the industry selected in the chart if one is selected
    default_ind = sel_ind if sel_ind != "All Industries" and sel_ind is not None else (metrics['Industry'] if metrics else None)
    
    if not INDUSTRY_BENCHMARKS_DF.empty:
        inds_list = sorted(INDUSTRY_BENCHMARKS_DF['Business_Industry'].unique())
        try:
            d_ix = inds_list.index(default_ind) if default_ind in inds_list else 0
        except ValueError:
            d_ix = 0
        p_ind = st.selectbox("3.1 Select Industry", inds_list, index=d_ix)
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
        # Pre-format as strings to avoid errors
        pdf['Projected Cases'] = pdf['Projected Cases'].map('{:,.1f}'.format)
        pdf['% of Total'] = pdf['% of Total'].map('{:.1f}%'.format)
        
        st.dataframe(
            pdf,
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
