import streamlit as st
import pandas as pd
import numpy as np
import re
import base64
from io import BytesIO
import logging
import openpyxl
from openpyxl.styles import Alignment, Border, Side, Font
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to process the data
def process_data(positions_file, nfo_bhav_file, bfo_bhav_file, expiry_nfo, expiry_bfo, include_settlement_nfo, include_settlement_bfo):
    try:
        # Read the positions file
        df = pd.read_csv(positions_file)
        logger.info("Positions CSV loaded successfully")

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

        # Initialize variables
        total_settlement_pnl_nfo = 0
        total_settlement_pnl_bfo = 0
        df_bhav_nfo = pd.DataFrame()
        df_bhav_bfo = pd.DataFrame()

        # Process NFO Bhavcopy if settlement is included
        if include_settlement_nfo:
            if nfo_bhav_file is None:
                raise ValueError("NFO bhavcopy file is required when settlement PNL for NFO is enabled")
            df_bhav_nfo = pd.read_csv(nfo_bhav_file)
            logger.info("NFO bhavcopy loaded successfully")

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

        # Process BFO Bhavcopy if settlement is included
        if include_settlement_bfo:
            if bfo_bhav_file is None:
                raise ValueError("BFO bhavcopy file is required when settlement PNL for BFO is enabled")
            df_bhav_bfo = pd.read_csv(bfo_bhav_file)
            logger.info("BFO bhavcopy loaded successfully")

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

        # Calculate Settlement PNL for NFO (if enabled)
        if include_settlement_nfo:
            df_nfo["Calculated_Settlement_PNL"] = np.select(
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
            df_nfo["Matching_Settlement"] = df_nfo["Unrealized Profit"] == df_nfo["Calculated_Settlement_PNL"]
            df_nfo["Matching_Settlement"] = df_nfo["Matching_Settlement"].replace({True: "TRUE", False: ""})
            total_settlement_pnl_nfo = df_nfo["Calculated_Settlement_PNL"].fillna(0).sum()
        else:
            df_nfo["Calculated_Settlement_PNL"] = 0
            df_nfo["Matching_Settlement"] = ""

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

        # Calculate Settlement PNL for BFO (if enabled)
        if include_settlement_bfo:
            long_condition = df_bfo["Net Qty"] > 0
            df_bfo.loc[long_condition, "Calculated_Settlement_PNL"] = (
                (df_bfo["Close Price"] - df_bfo["Buy Avg Price"]) * abs(df_bfo["Net Qty"])
            )
            short_condition = df_bfo["Net Qty"] < 0
            df_bfo.loc[short_condition, "Calculated_Settlement_PNL"] = (
                (df_bfo["Sell Avg Price"] - df_bfo["Close Price"]) * abs(df_bfo["Net Qty"])
            )
            df_bfo.loc[df_bfo["Net Qty"] == 0, "Calculated_Settlement_PNL"] = 0
            df_bfo["Matching_Settlement"] = df_bfo["Unrealized Profit"] == df_bfo["Calculated_Settlement_PNL"]
            df_bfo["Matching_Settlement"] = df_bfo["Matching_Settlement"].replace({True: "TRUE", False: ""})
            total_settlement_pnl_bfo = df_bfo["Calculated_Settlement_PNL"].fillna(0).sum()
        else:
            df_bfo["Calculated_Settlement_PNL"] = 0
            df_bfo["Matching_Settlement"] = ""

        total_realized_pnl_bfo = df_bfo["Calculated_Realized_PNL"].fillna(0).sum()

        # Overall totals
        overall_realized = total_realized_pnl_nfo + total_realized_pnl_bfo
        overall_settlement = total_settlement_pnl_nfo + total_settlement_pnl_bfo
        grand_total = overall_realized + overall_settlement

        logger.info("Data processing completed successfully")
        return {
            "df_nfo": df_nfo,
            "df_bfo": df_bfo,
            "total_realized_nfo": total_realized_pnl_nfo,
            "total_settlement_nfo": total_settlement_pnl_nfo,
            "total_realized_bfo": total_realized_pnl_bfo,
            "total_settlement_bfo": total_settlement_pnl_bfo,
            "overall_realized": overall_realized,
            "overall_settlement": overall_settlement,
            "grand_total": grand_total
        }
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        raise

# Function to get Excel download link with centralized formatting using openpyxl
def get_excel_download_link(df, filename):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='PNL Data')
        worksheet = writer.sheets['PNL Data']
        
        # Center align all data
        for row in worksheet.rows:
            for cell in row:
                cell.alignment = Alignment(horizontal='center')
        
        # Apply good formatting (bold headers, borders, etc.)
        for cell in worksheet[1]:  # Row 1 is the header row
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = openpyxl.styles.PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
        
        # Add borders
        for row in worksheet.rows:
            for cell in row:
                cell.border = Border(left=Side(style='thin'),
                                   right=Side(style='thin'),
                                   top=Side(style='thin'),
                                   bottom=Side(style='thin'))
    
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx" class="text-blue-600 hover:text-blue-800 font-semibold">Download {filename}.xlsx</a>'

# Streamlit App Configuration
st.set_page_config(
    page_title="Ultimate Financial PNL Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Tailwind CSS and Animations
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f3f4f6, #e5e7eb);
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 1rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #3b82f6, #60a5fa);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #2563eb, #3b82f6);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .stDateInput input {
        border: 2px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 0.5rem;
        background: #ffffff;
    }
    .stFileUploader button {
        background: linear-gradient(45deg, #10b981, #34d399);
        color: white;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
    .stFileUploader button:hover {
        background: linear-gradient(45deg, #059669, #10b981);
    }
    .stCheckbox label {
        font-size: 1rem;
        color: #1f2937;
    }
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-label {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #e5e7eb;
    }
    .insights-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1.5rem;
    }
    .chart-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .header-text {
        font-size: 2.25rem;
        font-weight: 800;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader-text {
        font-size: 1.25rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# Theme toggle with improved styling
with st.sidebar:
    st.markdown('<h2 class="text-lg font-bold text-gray-800 dark:text-gray-200 mb-4">🌗 Theme</h2>', unsafe_allow_html=True)
    theme = st.selectbox("Select Theme", ["Light", "Dark"], key="theme_select", help="Switch between light and dark themes")
    if theme == "Dark":
        st.markdown("""
            <style>
            body {
                background: linear-gradient(135deg, #1f2937, #374151);
                color: #f3f4f6;
            }
            .sidebar .sidebar-content {
                background: #374151;
            }
            .metric-card, .insights-box, .chart-container {
                background: #4b5563;
            }
            .metric-label {
                color: #d1d5db;
            }
            .header-text, .subheader-text {
                color: #f3f4f6;
            }
            .stTabs [data-baseweb="tab"] {
                color: #f3f4f6;
            }
            .stTabs [data-baseweb="tab"]:hover {
                background: #6b7280;
            }
            </style>
        """, unsafe_allow_html=True)

# Sidebar for inputs with improved UX
with st.sidebar:
    st.markdown('<h2 class="text-lg font-bold text-gray-800 dark:text-gray-200 mb-4">📁 Upload Data</h2>', unsafe_allow_html=True)
    positions_file = st.file_uploader("Positions CSV", type="csv", help="Upload VS20 22 AUG 2025 POSITIONS(EOD).csv", key="positions_upload")
    
    include_settlement_nfo = st.checkbox("Include Settlement PNL for NFO", value=True, help="Uncheck to exclude settlement PNL for NFO", key="nfo_settlement")
    include_settlement_bfo = st.checkbox("Include Settlement PNL for BFO", value=True, help="Uncheck to exclude settlement PNL for BFO", key="bfo_settlement")

    if include_settlement_nfo:
        nfo_bhav_file = st.file_uploader("NFO Bhavcopy", type="csv", help="Upload op220825.csv", key="nfo_upload")
    else:
        nfo_bhav_file = None
        st.info("NFO bhavcopy not required when settlement PNL for NFO is disabled.")

    if include_settlement_bfo:
        bfo_bhav_file = st.file_uploader("BFO Bhavcopy", type="csv", help="Upload MS_20250822-01.csv", key="bfo_upload")
    else:
        bfo_bhav_file = None
        st.info("BFO bhavcopy not required when settlement PNL for BFO is disabled.")

    st.markdown('<h2 class="text-lg font-bold text-gray-800 dark:text-gray-200 mb-4">📅 Expiry Dates</h2>', unsafe_allow_html=True)
    expiry_nfo = st.date_input("NFO Expiry Date", value=datetime(2025, 8, 28), help="Format: YYYY-MM-DD", key="nfo_expiry", disabled=not include_settlement_nfo)
    expiry_bfo = st.date_input("BFO Expiry Date", value=datetime(2025, 8, 26), help="Format: YYYY-MM-DD", key="bfo_expiry", disabled=not include_settlement_bfo)

    process_button = st.button("🚀 Process Data", key="process_button")

# Main content with enhanced typography and layout
st.markdown('<h1 class="header-text">📈 Ultimate Financial PNL Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Analyze your portfolio with real-time insights, interactive charts, and comprehensive data exports for smarter trading decisions.</p>', unsafe_allow_html=True)

if process_button:
    if positions_file:
        if (include_settlement_nfo and nfo_bhav_file is None) or (include_settlement_bfo and bfo_bhav_file is None):
            st.error("⚠️ Please upload all required CSV files for enabled settlement calculations.")
        else:
            try:
                with st.spinner("🔄 Processing your data..."):
                    results = process_data(positions_file, nfo_bhav_file, bfo_bhav_file, expiry_nfo, expiry_bfo, include_settlement_nfo, include_settlement_bfo)

                # Key Metrics Section with improved grid layout
                st.markdown('<h2 class="text-xl font-bold text-gray-800 dark:text-gray-200 mt-8 mb-4">🔑 Key Financial Metrics</h2>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3, gap="medium")
                with col1:
                    color = '#34D399' if results["overall_realized"] > 0 else '#F87171' if results["overall_realized"] < 0 else '#9CA3AF'
                    st.markdown(f'<div class="metric-card"><p class="metric-label">Overall Realized PNL</p><p class="metric-value" style="color:{color};">₹{results["overall_realized"]:,.2f}</p></div>', unsafe_allow_html=True)
                if include_settlement_nfo or include_settlement_bfo:
                    with col2:
                        color = '#34D399' if results["overall_settlement"] > 0 else '#F87171' if results["overall_settlement"] < 0 else '#9CA3AF'
                        st.markdown(f'<div class="metric-card"><p class="metric-label">Overall Settlement PNL</p><p class="metric-value" style="color:{color};">₹{results["overall_settlement"]:,.2f}</p></div>', unsafe_allow_html=True)
                with col3:
                    color = '#34D399' if results["grand_total"] > 0 else '#F87171' if results["grand_total"] < 0 else '#9CA3AF'
                    st.markdown(f'<div class="metric-card"><p class="metric-label">Grand Total PNL</p><p class="metric-value" style="color:{color};">₹{results["grand_total"]:,.2f}</p></div>', unsafe_allow_html=True)

                col4, col5, col6, col7 = st.columns(4, gap="medium")
                with col4:
                    color = '#34D399' if results["total_realized_nfo"] > 0 else '#F87171' if results["total_realized_nfo"] < 0 else '#9CA3AF'
                    st.markdown(f'<div class="metric-card"><p class="metric-label">NFO Realized PNL</p><p class="metric-value" style="color:{color};">₹{results["total_realized_nfo"]:,.2f}</p></div>', unsafe_allow_html=True)
                if include_settlement_nfo:
                    with col5:
                        color = '#34D399' if results["total_settlement_nfo"] > 0 else '#F87171' if results["total_settlement_nfo"] < 0 else '#9CA3AF'
                        st.markdown(f'<div class="metric-card"><p class="metric-label">NFO Settlement PNL</p><p class="metric-value" style="color:{color};">₹{results["total_settlement_nfo"]:,.2f}</p></div>', unsafe_allow_html=True)
                with col6:
                    color = '#34D399' if results["total_realized_bfo"] > 0 else '#F87171' if results["total_realized_bfo"] < 0 else '#9CA3AF'
                    st.markdown(f'<div class="metric-card"><p class="metric-label">BFO Realized PNL</p><p class="metric-value" style="color:{color};">₹{results["total_realized_bfo"]:,.2f}</p></div>', unsafe_allow_html=True)
                if include_settlement_bfo:
                    with col7:
                        color = '#34D399' if results["total_settlement_bfo"] > 0 else '#F87171' if results["total_settlement_bfo"] < 0 else '#9CA3AF'
                        st.markdown(f'<div class="metric-card"><p class="metric-label">BFO Settlement PNL</p><p class="metric-value" style="color:{color};">₹{results["total_settlement_bfo"]:,.2f}</p></div>', unsafe_allow_html=True)

                # Comprehensive Results and Summary Excel Download
                st.markdown('<h2 class="text-xl font-bold text-gray-800 dark:text-gray-200 mt-8 mb-4">📥 Download Results</h2>', unsafe_allow_html=True)
                col_download1 = st.columns(1)[0]  # Single column for single file download
                with col_download1:
                    # Create DataFrame for the requested format with dynamically calculated values
                    pnl_breakdown_df = pd.DataFrame({
                        'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        'NFO Realized PNL': [results["total_realized_nfo"]],
                        'NFO Settlement PNL': [results["total_settlement_nfo"] if include_settlement_nfo else 0],
                        'Nfo total': [results["total_realized_nfo"] + (results["total_settlement_nfo"] if include_settlement_nfo else 0)],
                        'BFO Realized PNL': [results["total_realized_bfo"]],
                        'BFO Settlement PNL': [results["total_settlement_bfo"] if include_settlement_bfo else 0],
                        'BFO total': [results["total_realized_bfo"] + (results["total_settlement_bfo"] if include_settlement_bfo else 0)],
                        'Grand Total PNL': [results["grand_total"]]
                    })
                    st.markdown(get_excel_download_link(pnl_breakdown_df, "A19_data"), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Error processing data: {str(e)}")
                st.info("Please ensure all uploaded CSV files have the correct columns and data format. Check the log for details.")
    else:
        st.error("⚠️ Please upload the positions CSV file to proceed.")
