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
        df = pd.read_csv("app_raw_data.csv")
        df.columns = df.columns.str.strip()
        
        rename_dict = {}
        for req_col, internal in COLUMN_MAP.items():
            if req_col in df.columns:
                rename_dict[req_col] = internal
            else:
                # Fallback fuzzy match
                for col in df.columns:
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
        st.error("Data file 'app_raw_data.csv' not found.")
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
        return None
        
    metrics = {
        'Industry': industry, 'Subscribers': subs, 
        'Customer_Since': client_row['Customer_Since'],
        'Client_Case_Totals': {col: client_row[col] for col in CASE_COLUMNS},
        'Industry_Case_Rates': {col: ind_row[f"{col}_Rate"] for col in CASE_COLUMNS}
    }
    
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
        metrics['Util_Comparison'] = pd.DataFrame(util_comp)
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
            bd_df = bd_df[bd_df['Cases'] > 0]
            
            if not bd_df.empty:
                tot = bd_df['Cases'].sum()
                # Format as string directly to avoid syntax errors in column_config
                bd_df['% of Total'] = (bd_df['Cases'] / tot * 100).map('{:.1f}%'.format)
                
                st.subheader("Client Case Breakdown")
                st.dataframe(
                    bd_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No cases recorded.")
        
        # --- CHART: Case Rate Comparison ---
        st.write("")
        st.subheader("Client vs. Industry: Case Rate Comparison")
        
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
        st.markdown(f'<h3 style="color:{BRAND_COLOR_DARK};">Utilization Analysis</h3>', unsafe_allow_html=True)
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
                # Better: use the parallel series diff_values
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
st.markdown(f'<h2 style="color:{BRAND_COLOR_BLUE};">2. Correlation: Utilization vs. Case Load</h2>', unsafe_allow_html=True)
st.write("Comparing **Total Utilization** against **Total Case Load**. Outliers (top 5%) are removed.")

inds = sorted(RAW_DATA_DF['Business_Industry'].dropna().unique().tolist())
sel_ind = st.selectbox("Filter Chart by Industry", ["All Industries"] + inds)

plot_df = RAW_DATA_DF[RAW_DATA_DF['Subscribers'] > 0].copy()
if sel_ind != "All Industries":
    plot_df = plot_df[plot_df['Business_Industry'] == sel_ind]

if not plot_df.empty:
    util_cap = plot_df['Utilization_Per_Subscriber'].quantile(0.95)
    plot_df_filtered = plot_df[plot_df['Utilization_Per_Subscriber'] <= util_cap].copy()
else:
    plot_df_filtered = pd.DataFrame()

if sel_id != "Select here...":
    sel_row = RAW_DATA_DF[RAW_DATA_DF['AccountID'] == str(sel_id)]
    if not sel_row.empty and str(sel_id) not in plot_df_filtered['AccountID'].values:
             plot_df_filtered = pd.concat([plot_df_filtered, sel_row])

    plot_df_filtered['Client_Type'] = plot_df_filtered['AccountID'].apply(lambda x: 'Selected Client' if str(x) == str(sel_id) else 'Other Clients')
    cmap = {'Selected Client': BRAND_COLOR_ORANGE, 'Other Clients': BRAND_COLOR_BLUE}
else:
    plot_df_filtered['Client_Type'] = 'All Clients'
    cmap = {'All Clients': BRAND_COLOR_BLUE}

if not plot_df_filtered.empty:
    fig = px.scatter(
        plot_df_filtered, x="Utilization_Per_Subscriber", y="Cases_Per_Subscriber",
        color="Client_Type", color_discrete_map=cmap,
        hover_data=["AccountID", "Business_Industry", "Subscribers"],
        labels={"Utilization_Per_Subscriber": "Total Utilization/Subscriber", "Cases_Per_Subscriber": "Total Cases/Subscriber"},
        height=500
    )
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available.")

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
