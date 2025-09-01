import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to process the data
def process_data(positions_file, nfo_bhav_file, bfo_bhav_file, expiry_nfo, expiry_bfo, include_unrealized_nfo, include_unrealized_bfo):
    try:
        # Read the files
        df = pd.read_csv(positions_file)
        df_bhav_nfo = pd.read_csv(nfo_bhav_file)
        df_bhav_bfo = pd.read_csv(bfo_bhav_file)
        logger.info("CSV files loaded successfully")

        # Verify required columns
        required_columns = ['Exchange', 'Symbol', 'Net Qty', 'Buy Avg Price', 'Sell Avg Price', 'Sell Qty', 'Buy Qty', 'Realized Profit', 'Unrealized Profit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in positions file: {missing_columns}")

        # Process Symbol for NFO and BFO
        mask = df["Exchange"].isin(["NFO", "BFO"])
        if 'Symbol' in df.columns:
            df.loc[mask, "Symbol"] = df.loc[mask, "Symbol"].astype(str).str[-5:] + df.loc[mask, "Symbol"].astype(str).str[-8:-6]
        else:
            raise KeyError("Symbol column not found in positions file")

        # Split into NFO and BFO
        df_nfo = df[df["Exchange"] == "NFO"].copy()
        df_bfo = df[df["Exchange"] == "BFO"].copy()
        logger.info("Data split into NFO and BFO")

        # Process NFO Bhavcopy
        if 'CONTRACT_D' not in df_bhav_nfo.columns:
            raise ValueError("CONTRACT_D column not found in NFO bhavcopy")
        df_bhav_nfo["Date"] = df_bhav_nfo["CONTRACT_D"].str.extract(r'(\d{2}-[A-Z]{3}-\d{4})')
        df_bhav_nfo["Symbol"] = df_bhav_nfo["CONTRACT_D"].str.extract(r'^(.*?)(\d{2}-[A-Z]{3}-\d{4})')[0]
        df_bhav_nfo["Strike_Type"] = df_bhav_nfo["CONTRACT_D"].str.extract(r'(PE\d+|CE\d+)$')
        df_bhav_nfo["Date"] = pd.to_datetime(df_bhav_nfo["Date"], format="%d-%b-%Y")
        df_bhav_nfo["Strike_Type"] = df_bhav_nfo["Strike_Type"].str.replace(
            r'^(PE|CE)(\d+)$', r'\2\1', regex=True
        )

        target_symbol = "OPTIDXNIFTY"
        df_bhav_nfo = df_bhav_nfo[
            (df_bhav_nfo["Date"] == pd.to_datetime(expiry_nfo)) &
            (df_bhav_nfo["Symbol"] == target_symbol)
        ]

        df_nfo["Strike_Type"] = df_nfo["Symbol"].str.extract(r'(\d+[A-Z]{2})$')
        df_nfo['Strike'] = df_nfo['Strike_Type'].str[:-2].astype(float, errors='ignore')
        df_nfo['Option_Type'] = df_nfo['Strike_Type'].str[-2:]

        # Merge to get SETTLEMENT
        df_nfo = df_nfo.merge(
            df_bhav_nfo[["Strike_Type", "SETTLEMENT"]],
            on="Strike_Type",
            how="left"
        )
        logger.info("NFO data merged with bhavcopy")

        # Process BFO Bhavcopy
        if 'Market Summary Date' not in df_bhav_bfo.columns or 'Expiry Date' not in df_bhav_bfo.columns or 'Series Code' not in df_bhav_bfo.columns:
            raise ValueError("Required columns missing in BFO bhavcopy")
        df_bhav_bfo["Date"] = pd.to_datetime(df_bhav_bfo["Market Summary Date"], format="%d %b %Y", errors="coerce")
        df_bhav_bfo["Expiry Date"] = pd.to_datetime(df_bhav_bfo["Expiry Date"], format="%d %b %Y", errors="coerce")
        df_bhav_bfo["Symbols"] = df_bhav_bfo["Series Code"].astype(str).str[-7:]

        df_bhav_bfo = df_bhav_bfo[
            (df_bhav_bfo["Expiry Date"] == pd.to_datetime(expiry_bfo))
        ]

        df_bfo["Symbol"] = df_bfo["Symbol"].astype(str).str.strip()
        df_bhav_bfo["Symbols"] = df_bhav_bfo["Symbols"].astype(str).str.strip()

        bhav_mapping = df_bhav_bfo.drop_duplicates(subset="Symbols", keep="last").set_index("Symbols")["Close Price"]
        df_bfo["Close Price"] = df_bfo["Symbol"].map(bhav_mapping)
        logger.info("BFO data processed")

        # Calculate Realized PNL for NFO
        conditions = [
            df_nfo["Net Qty"] == 0,
            df_nfo["Net Qty"] > 0,
            df_nfo["Net Qty"] < 0
        ]
        choices = [
            (df_nfo["Sell Avg Price"] - df_nfo["Buy Avg Price"]) * df_nfo["Sell Qty"],
            (df_nfo["Sell Avg Price"] - df_nfo["Buy Avg Price"]) * df_nfo["Sell Qty"],
            (df_nfo["Sell Avg Price"] - df_nfo["Buy Avg Price"]) * df_nfo["Buy Qty"]
        ]
        df_nfo["Calculated_Realized_PNL"] = np.select(conditions, choices, default=0)
        df_nfo["Matching_Realized"] = df_nfo["Realized Profit"] == df_nfo["Calculated_Realized_PNL"]
        df_nfo["Matching_Realized"] = df_nfo["Matching_Realized"].replace({True: "TRUE", False: ""})

        # Calculate Unrealized PNL for NFO (if enabled)
        total_unrealized_pnl_nfo = 0
        if include_unrealized_nfo:
            df_nfo["Calculated_Unrealized_PNL"] = np.select(
                [
                    df_nfo["Net Qty"] > 0,
                    df_nfo["Net Qty"] < 0
                ],
                [
                    (df_nfo["SETTLEMENT"] - df_nfo["Buy Avg Price"]) * abs(df_nfo["Net Qty"]),
                    (df_nfo["Sell Avg Price"] - df_nfo["SETTLEMENT"]) * abs(df_nfo["Net Qty"])
                ],
                default=0
            )
            df_nfo["Matching_Unrealized"] = df_nfo["Unrealized Profit"] == df_nfo["Calculated_Unrealized_PNL"]
            df_nfo["Matching_Unrealized"] = df_nfo["Matching_Unrealized"].replace({True: "TRUE", False: ""})
            total_unrealized_pnl_nfo = df_nfo["Calculated_Unrealized_PNL"].fillna(0).sum()
        else:
            df_nfo["Calculated_Unrealized_PNL"] = 0
            df_nfo["Matching_Unrealized"] = ""

        total_realized_pnl_nfo = df_nfo["Calculated_Realized_PNL"].fillna(0).sum()

        # Calculate Realized PNL for BFO
        conditions_bfo = [
            df_bfo["Net Qty"] == 0,
            df_bfo["Net Qty"] > 0,
            df_bfo["Net Qty"] < 0
        ]
        choices_bfo = [
            (df_bfo["Sell Avg Price"] - df_bfo["Buy Avg Price"]) * df_bfo["Sell Qty"],
            (df_bfo["Sell Avg Price"] - df_bfo["Buy Avg Price"]) * df_bfo["Sell Qty"],
            (df_bfo["Sell Avg Price"] - df_bfo["Buy Avg Price"]) * df_bfo["Buy Qty"]
        ]
        df_bfo["Calculated_Realized_PNL"] = np.select(conditions_bfo, choices_bfo, default=0)
        df_bfo["Matching_Realized"] = df_bfo["Realized Profit"] == df_bfo["Calculated_Realized_PNL"]
        df_bfo["Matching_Realized"] = df_bfo["Matching_Realized"].replace({True: "TRUE", False: ""})

        # Calculate Unrealized PNL for BFO (if enabled)
        total_unrealized_pnl_bfo = 0
        if include_unrealized_bfo:
            long_condition = df_bfo["Net Qty"] > 0
            df_bfo.loc[long_condition, "Calculated_Unrealized_PNL"] = (
                (df_bfo["Close Price"] - df_bfo["Buy Avg Price"]) * abs(df_bfo["Net Qty"])
            )
            short_condition = df_bfo["Net Qty"] < 0
            df_bfo.loc[short_condition, "Calculated_Unrealized_PNL"] = (
                (df_bfo["Sell Avg Price"] - df_bfo["Close Price"]) * abs(df_bfo["Net Qty"])
            )
            df_bfo.loc[df_bfo["Net Qty"] == 0, "Calculated_Unrealized_PNL"] = 0
            df_bfo["Matching_Unrealized"] = df_bfo["Unrealized Profit"] == df_bfo["Calculated_Unrealized_PNL"]
            df_bfo["Matching_Unrealized"] = df_bfo["Matching_Unrealized"].replace({True: "TRUE", False: ""})
            total_unrealized_pnl_bfo = df_bfo["Calculated_Unrealized_PNL"].fillna(0).sum()
        else:
            df_bfo["Calculated_Unrealized_PNL"] = 0
            df_bfo["Matching_Unrealized"] = ""

        total_realized_pnl_bfo = df_bfo["Calculated_Realized_PNL"].fillna(0).sum()

        # Overall totals
        overall_realized = total_realized_pnl_nfo + total_realized_pnl_bfo
        overall_unrealized = total_unrealized_pnl_nfo + total_unrealized_pnl_bfo
        grand_total = overall_realized + overall_unrealized

        # Aggregated data for insights
        if 'Symbol' not in df_nfo.columns:
            df_nfo['Symbol'] = df_nfo['Strike_Type']
        if 'Symbol' not in df_bfo.columns:
            df_bfo['Symbol'] = df_bfo['Symbol']

        nfo_agg = df_nfo.groupby('Symbol').agg({
            'Calculated_Realized_PNL': 'sum',
            'Calculated_Unrealized_PNL': 'sum'
        }).reset_index()
        nfo_agg['Total_PNL'] = nfo_agg['Calculated_Realized_PNL'] + nfo_agg['Calculated_Unrealized_PNL']

        bfo_agg = df_bfo.groupby('Symbol').agg({
            'Calculated_Realized_PNL': 'sum',
            'Calculated_Unrealized_PNL': 'sum'
        }).reset_index()
        bfo_agg['Total_PNL'] = bfo_agg['Calculated_Realized_PNL'] + bfo_agg['Calculated_Unrealized_PNL']

        # Top gainers/losers
        top_gainers_nfo = nfo_agg.nlargest(5, 'Total_PNL')
        top_losers_nfo = nfo_agg.nsmallest(5, 'Total_PNL')
        top_gainers_bfo = bfo_agg.nlargest(5, 'Total_PNL')
        top_losers_bfo = bfo_agg.nsmallest(5, 'Total_PNL')

        logger.info("Data processing completed successfully")
        return {
            "df_nfo": df_nfo,
            "df_bfo": df_bfo,
            "nfo_agg": nfo_agg,
            "bfo_agg": bfo_agg,
            "top_gainers_nfo": top_gainers_nfo,
            "top_losers_nfo": top_losers_nfo,
            "top_gainers_bfo": top_gainers_bfo,
            "top_losers_bfo": top_losers_bfo,
            "total_realized_nfo": total_realized_pnl_nfo,
            "total_unrealized_nfo": total_unrealized_pnl_nfo,
            "total_realized_bfo": total_realized_pnl_bfo,
            "total_unrealized_bfo": total_unrealized_pnl_bfo,
            "overall_realized": overall_realized,
            "overall_unrealized": overall_unrealized,
            "grand_total": grand_total
        }
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        raise

# Function to get CSV download link
def get_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

# Function to get color based on value
def get_color(value):
    return '#4CAF50' if value > 0 else '#F44336' if value < 0 else '#757575'

# Streamlit App
st.set_page_config(page_title="Ultimate Financial PNL Dashboard", layout="wide", page_icon="📊")

# Custom CSS for enhanced UI/UX
st.markdown("""
    <style>
    .main {background: linear-gradient(to bottom, #f8f9fa, #e9ecef); font-family: 'Roboto', sans-serif;}
    .stButton>button {background: linear-gradient(45deg, #007bff, #00d4ff); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: bold; transition: all 0.3s;}
    .stButton>button:hover {background: linear-gradient(45deg, #0056b3, #0096cc); transform: scale(1.05);}
    .stDateInput>div>div>input {border-radius: 8px; padding: 10px; border: 2px solid #007bff;}
    .stFileUploader>div>button {background: linear-gradient(45deg, #28a745, #34c759); color: white; border-radius: 8px; padding: 10px;}
    .stCheckbox label {font-size: 16px; color: #333;}
    .metric-card {background: white; padding: 20px; border-radius: 12px; box-shadow: 0 6px 12px rgba(0,0,0,0.1); text-align: center; transition: transform 0.3s;}
    .metric-card:hover {transform: translateY(-5px);}
    .metric-label {font-size: 18px; color: #6c757d; margin-bottom: 10px;}
    .metric-value {font-size: 28px; font-weight: bold;}
    .stTabs [data-baseweb="tab"] {font-size: 16px; font-weight: bold; padding: 10px 20px;}
    .insights-box {background: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px;}
    footer {visibility: hidden;}
    .chart-container {padding: 20px; background: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Theme toggle
theme = st.sidebar.selectbox("🌗 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        .main {background: linear-gradient(to bottom, #343a40, #495057); color: #f8f9fa;}
        .metric-card {background: #495057;}
        .metric-label {color: #ced4da;}
        .insights-box {background: #495057;}
        .chart-container {background: #343a40;}
        </style>
    """, unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("📁 Upload Your Data")
    positions_file = st.file_uploader("Positions CSV", type="csv", help="Upload VS20 22 AUG 2025 POSITIONS(EOD).csv")
    nfo_bhav_file = st.file_uploader("NFO Bhavcopy", type="csv", help="Upload op220825.csv")
    bfo_bhav_file = st.file_uploader("BFO Bhavcopy", type="csv", help="Upload MS_20250822-01.csv")

    st.header("📅 Select Expiry Dates")
    expiry_nfo = st.date_input("NFO Expiry Date", value=datetime(2025, 8, 28), help="Format: YYYY-MM-DD")
    expiry_bfo = st.date_input("BFO Expiry Date", value=datetime(2025, 8, 26), help="Format: YYYY-MM-DD")

    st.header("⚙️ Calculation Options")
    include_unrealized_nfo = st.checkbox("Include Unrealized PNL for NFO", value=True, help="Uncheck to exclude unrealized PNL calculations for NFO")
    include_unrealized_bfo = st.checkbox("Include Unrealized PNL for BFO", value=True, help="Uncheck to exclude unrealized PNL calculations for BFO")

    process_button = st.button("🚀 Process Data")

# Main content
st.title("📈 Ultimate Financial PNL Dashboard")
st.markdown("Analyze your portfolio with ease. Upload files, select dates, and choose whether to include unrealized PNL for NFO and BFO. Explore interactive visualizations to make informed trading decisions.")

if process_button:
    if positions_file and nfo_bhav_file and bfo_bhav_file:
        try:
            with st.spinner("🔄 Processing your data..."):
                results = process_data(positions_file, nfo_bhav_file, bfo_bhav_file, expiry_nfo, expiry_bfo, include_unrealized_nfo, include_unrealized_bfo)

            # Key Metrics Section
            st.header("🔑 Key Financial Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                color = get_color(results["overall_realized"])
                st.markdown(f'<div class="metric-card"><p class="metric-label">Overall Realized PNL</p><p class="metric-value" style="color:{color};">₹{results["overall_realized"]:,.2f}</p></div>', unsafe_allow_html=True)
            with col2:
                color = get_color(results["overall_unrealized"])
                st.markdown(f'<div class="metric-card"><p class="metric-label">Overall Unrealized PNL</p><p class="metric-value" style="color:{color};">₹{results["overall_unrealized"]:,.2f}</p></div>', unsafe_allow_html=True)
            with col3:
                color = get_color(results["grand_total"])
                st.markdown(f'<div class="metric-card"><p class="metric-label">Grand Total PNL</p><p class="metric-value" style="color:{color};">₹{results["grand_total"]:,.2f}</p></div>', unsafe_allow_html=True)

            col4, col5, col6, col7 = st.columns(4)
            with col4:
                color = get_color(results["total_realized_nfo"])
                st.markdown(f'<div class="metric-card"><p class="metric-label">NFO Realized PNL</p><p class="metric-value" style="color:{color};">₹{results["total_realized_nfo"]:,.2f}</p></div>', unsafe_allow_html=True)
            with col5:
                color = get_color(results["total_unrealized_nfo"])
                st.markdown(f'<div class="metric-card"><p class="metric-label">NFO Unrealized PNL</p><p class="metric-value" style="color:{color};">₹{results["total_unrealized_nfo"]:,.2f}</p></div>', unsafe_allow_html=True)
            with col6:
                color = get_color(results["total_realized_bfo"])
                st.markdown(f'<div class="metric-card"><p class="metric-label">BFO Realized PNL</p><p class="metric-value" style="color:{color};">₹{results["total_realized_bfo"]:,.2f}</p></div>', unsafe_allow_html=True)
            with col7:
                color = get_color(results["total_unrealized_bfo"])
                st.markdown(f'<div class="metric-card"><p class="metric-label">BFO Unrealized PNL</p><p class="metric-value" style="color:{color};">₹{results["total_unrealized_bfo"]:,.2f}</p></div>', unsafe_allow_html=True)

            # Interactive Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 NFO Data", "📊 BFO Data", "📈 Visualizations", "💡 Insights"])

            with tab1:
                st.subheader("NFO Processed Data")
                st.dataframe(results["df_nfo"].style.background_gradient(cmap='RdYlGn', subset=['Calculated_Realized_PNL', 'Calculated_Unrealized_PNL']).format({"Calculated_Realized_PNL": "₹{:.2f}", "Calculated_Unrealized_PNL": "₹{:.2f}"}, na_rep="N/A"), use_container_width=True)
                st.markdown(get_csv_download_link(results["df_nfo"], "pos_nfo.csv"), unsafe_allow_html=True)

                with st.expander("Aggregated NFO by Symbol"):
                    st.dataframe(results["nfo_agg"].style.background_gradient(cmap='RdYlGn', subset=['Total_PNL']).format({"Total_PNL": "₹{:.2f}"}, na_rep="N/A"), use_container_width=True)

            with tab2:
                st.subheader("BFO Processed Data")
                st.dataframe(results["df_bfo"].style.background_gradient(cmap='RdYlGn', subset=['Calculated_Realized_PNL', 'Calculated_Unrealized_PNL']).format({"Calculated_Realized_PNL": "₹{:.2f}", "Calculated_Unrealized_PNL": "₹{:.2f}"}, na_rep="N/A"), use_container_width=True)
                st.markdown(get_csv_download_link(results["df_bfo"], "pos_bfo.csv"), unsafe_allow_html=True)

                with st.expander("Aggregated BFO by Symbol"):
                    st.dataframe(results["bfo_agg"].style.background_gradient(cmap='RdYlGn', subset=['Total_PNL']).format({"Total_PNL": "₹{:.2f}"}, na_rep="N/A"), use_container_width=True)

            with tab3:
                st.subheader("Interactive PNL Visualizations")
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)

                # PNL Breakdown Bar Chart
                pnl_data = pd.DataFrame({
                    "Category": ["NFO Realized", "NFO Unrealized", "BFO Realized", "BFO Unrealized"],
                    "PNL": [results["total_realized_nfo"], results["total_unrealized_nfo"],
                            results["total_realized_bfo"], results["total_unrealized_bfo"]]
                })
                fig_bar = px.bar(pnl_data, x="Category", y="PNL", color="Category",
                                 title="PNL Breakdown by Category",
                                 color_discrete_sequence=px.colors.qualitative.Bold,
                                 labels={"PNL": "Profit/Loss (₹)"},
                                 text_auto='.2s')
                fig_bar.update_layout(hovermode="x unified", showlegend=False, title_x=0.5)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Overall PNL Pie Chart (only if unrealized is included)
                if include_unrealized_nfo or include_unrealized_bfo:
                    overall_pnl = pd.DataFrame({
                        "Type": ["Realized", "Unrealized"],
                        "Value": [abs(results["overall_realized"]), abs(results["overall_unrealized"])]
                    })
                    fig_pie = px.pie(overall_pnl, values="Value", names="Type",
                                     title="Overall PNL Distribution (Absolute Values)",
                                     color_discrete_sequence=px.colors.qualitative.Pastel,
                                     hole=0.3)
                    fig_pie.update_traces(textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                # NFO Strike vs PNL Scatter (only if unrealized NFO is included)
                if include_unrealized_nfo and not results["df_nfo"].empty and 'Strike' in results["df_nfo"].columns:
                    fig_scatter_nfo = px.scatter(results["df_nfo"], x="Strike", y="Calculated_Unrealized_PNL",
                                                 color="Option_Type", size=abs(results["df_nfo"]["Net Qty"]),
                                                 title="NFO: Unrealized PNL vs Strike Price",
                                                 hover_data=["Symbol", "SETTLEMENT", "Net Qty"],
                                                 labels={"Calculated_Unrealized_PNL": "Unrealized PNL (₹)"},
                                                 opacity=0.8)
                    fig_scatter_nfo.update_layout(transition_duration=500, title_x=0.5)
                    st.plotly_chart(fig_scatter_nfo, use_container_width=True)

                # Cumulative PNL Line Chart
                pnl_data['Cumulative'] = pnl_data['PNL'].cumsum()
                fig_line = px.line(pnl_data, x="Category", y="Cumulative",
                                   title="Cumulative PNL Across Categories",
                                   markers=True, line_shape="spline",
                                   labels={"Cumulative": "Cumulative PNL (₹)"})
                fig_line.update_traces(line=dict(width=4))
                st.plotly_chart(fig_line, use_container_width=True)

                # Heatmap for NFO PNL by Strike and Type (only if unrealized NFO is included)
                if include_unrealized_nfo and not results["df_nfo"].empty and 'Strike' in results["df_nfo"].columns:
                    pivot = results["df_nfo"].pivot_table(values='Calculated_Unrealized_PNL', index='Strike', columns='Option_Type', aggfunc='sum')
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns,
                        y=pivot.index,
                        colorscale='RdYlGn',
                        hoverongaps=False,
                        text=pivot.values,
                        texttemplate="₹%{text:,.2f}",
                        hovertemplate="Strike: %{y}<br>Option Type: %{x}<br>PNL: ₹%{z:,.2f}<extra></extra>"
                    ))
                    fig_heat.update_layout(title="NFO Unrealized PNL Heatmap by Strike and Option Type", title_x=0.5)
                    st.plotly_chart(fig_heat, use_container_width=True)

                # Treemap for Aggregated PNL
                treemap_data = pd.concat([
                    results["nfo_agg"][['Symbol', 'Total_PNL']].assign(Exchange='NFO'),
                    results["bfo_agg"][['Symbol', 'Total_PNL']].assign(Exchange='BFO')
                ])
                fig_tree = px.treemap(treemap_data, path=['Exchange', 'Symbol'], values='Total_PNL',
                                      color='Total_PNL', color_continuous_scale='RdYlGn',
                                      title="PNL Treemap by Exchange and Symbol")
                fig_tree.update_layout(title_x=0.5)
                st.plotly_chart(fig_tree, use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)

            with tab4:
                st.subheader("Key Insights & Recommendations")
                st.markdown('<div class="insights-box">', unsafe_allow_html=True)

                st.write("**Top 5 Gainers (NFO):**")
                st.dataframe(results["top_gainers_nfo"].style.background_gradient(cmap='Greens', subset=['Total_PNL']).format({"Total_PNL": "₹{:.2f}"}, na_rep="N/A"))

                st.write("**Top 5 Losers (NFO):**")
                st.dataframe(results["top_losers_nfo"].style.background_gradient(cmap='Reds', subset=['Total_PNL']).format({"Total_PNL": "₹{:.2f}"}, na_rep="N/A"))

                st.write("**Top 5 Gainers (BFO):**")
                st.dataframe(results["top_gainers_bfo"].style.background_gradient(cmap='Greens', subset=['Total_PNL']).format({"Total_PNL": "₹{:.2f}"}, na_rep="N/A"))

                st.write("**Top 5 Losers (BFO):**")
                st.dataframe(results["top_losers_bfo"].style.background_gradient(cmap='Reds', subset=['Total_PNL']).format({"Total_PNL": "₹{:.2f}"}, na_rep="N/A"))

                # Insights based on data
                if results["grand_total"] > 0:
                    st.success("🎉 Overall portfolio is in profit! Consider securing gains or scaling positions.")
                else:
                    st.warning("⚠️ Overall portfolio is in loss. Review high-loss positions and consider hedging strategies.")

                # Exposure analysis
                if not results["df_nfo"].empty:
                    net_exposure_nfo = results["df_nfo"]["Net Qty"].abs().sum()
                    st.info(f"**NFO Exposure**: {net_exposure_nfo:,.0f} units")
                if not results["df_bfo"].empty:
                    net_exposure_bfo = results["df_bfo"]["Net Qty"].abs().sum()
                    st.info(f"**BFO Exposure**: {net_exposure_bfo:,.0f} units")

                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error processing data: {str(e)}")
            st.info("Please ensure all uploaded CSV files have the correct columns and data format. Check the log for details.")
    else:
        st.error("⚠️ Please upload all required CSV files to proceed.")