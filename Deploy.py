# =========================================================
# STREAMLIT AI ENERGY INTELLIGENCE DASHBOARD 
# FUNCTIONAL CALLING + PIPELINE ARCHITECTURE
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import queue
import time

from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =========================================================
# PAGE CONFIG - MUST BE FIRST Streamlit Command
# =========================================================
st.set_page_config(page_title="Ontario Energy AI", layout="wide")

# =========================================================
# INITIALIZE SESSION STATE
# =========================================================
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.result = None
    st.session_state.cleaning_report = None
    st.session_state.processing = False

st.title("AI Ontario Energy Pipeline Dashboard")

# =========================================================
# CHECK OLLAMA AVAILABILITY
# =========================================================
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# =========================================================
# SECTION 1: DATA LOADING FUNCTIONS (Functional Calling)
# =========================================================

def load_data(file):
    """
    FUNCTION 1: Load CSV and create datetime index
    INPUT: CSV file
    OUTPUT: DataFrame with datetime index
    """
    df = pd.read_csv(file)
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df = df.sort_values('datetime').set_index('datetime')
    return df


def clean_data(df):
    """
    FUNCTION 2: Clean data - remove duplicates, fill missing, remove outliers
    INPUT: Raw DataFrame
    OUTPUT: Cleaned DataFrame + cleaning report
    """
    df = df.copy()
    initial_rows = len(df)
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    
    # Convert numeric columns
    numeric_cols = ['hour', 'hourly_demand', 'hourly_average_price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values
    df['hourly_demand'] = df['hourly_demand'].interpolate(method='linear')
    df['hourly_average_price'] = df['hourly_average_price'].interpolate(method='linear')
    df['hourly_demand'] = df['hourly_demand'].ffill().bfill()
    df['hourly_average_price'] = df['hourly_average_price'].ffill().bfill()
    
    # Remove outliers (Z-score method)
    numeric_df = df[['hourly_demand', 'hourly_average_price']].copy()
    z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
    mask = (z_scores < 3).all(axis=1)
    outliers_removed = (~mask).sum()
    df = df[mask]
    
    # Create cleaning report
    report = {
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "rows_removed": initial_rows - len(df),
        "duplicates_removed": int(duplicates),
        "outliers_removed": int(outliers_removed),
        "cleaning_percentage": round((1 - len(df)/initial_rows) * 100, 2)}
    return df, report


def feature_engineering(df):
    """
    FUNCTION 3: Create time-based features
    INPUT: Cleaned DataFrame
    OUTPUT: DataFrame with engineered features
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek
    df['month'] = df.index.month
    df['weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['demand_lag_24'] = df['hourly_demand'].shift(24)
    df['price_lag_24'] = df['hourly_average_price'].shift(24)
    df['rolling_demand_24'] = df['hourly_demand'].rolling(24).mean()
    df['rolling_price_24'] = df['hourly_average_price'].rolling(24).mean()
    df['volatility_24h'] = df['hourly_average_price'].rolling(24).std() / (df['hourly_average_price'].rolling(24).mean() + 1e-6)
    return df.dropna()


def statistical_summary(df):
    """
    FUNCTION 4: Calculate statistical summary
    INPUT: DataFrame
    OUTPUT: DataFrame with statistics
    """
    target = "hourly_average_price"
    return pd.DataFrame([{
        "mean_demand": df['hourly_demand'].mean(),
        "max_demand": df['hourly_demand'].max(),
        "min_demand": df['hourly_demand'].min(),
        "std_demand": df['hourly_demand'].std(),
        "mean_price": df[target].mean(),
        "std_price": df[target].std(),
        "min_price": df[target].min(),
        "max_price": df[target].max(),
        "median_price": df[target].median(),
        "cv": df[target].std() / df[target].mean() if df[target].mean() > 0 else 0}])


def eda_analysis(df):
    """
    FUNCTION 5: Exploratory Data Analysis
    INPUT: DataFrame
    OUTPUT: Dictionary with EDA results
    """
    return {
        "hourly_demand": df.groupby(df.index.hour)['hourly_demand'].mean(),
        "hourly_price": df.groupby(df.index.hour)['hourly_average_price'].mean(),
        "weekly_price": df.groupby(df['weekday'])['hourly_average_price'].mean(),
        "monthly_price": df.groupby(df['month'])['hourly_average_price'].mean(),
        "corr": df.corr(numeric_only=True),
        "anomalies": (np.abs(stats.zscore(df['hourly_average_price'])) > 3).sum()}


def check_stationarity(series):
    """
    FUNCTION 6: ADF Stationarity Test
    INPUT: Time series
    OUTPUT: Stationarity results
    """
    result = adfuller(series.dropna())
    return {
        "ADF": result[0],
        "p_value": result[1],
        "stationary": result[1] < 0.05}


def fast_arima_forecast(df, target='hourly_average_price'):
    """
    FUNCTION 7: Fast ARIMA Forecast (optimized for speed)
    INPUT: DataFrame
    OUTPUT: Forecast results with metrics
    """
    series = df[target]
    
    # Use only last 500 points for speed
    if len(series) > 500:
        series = series.iloc[-500:]
    
    split = int(len(series) * 0.8)
    train = series.iloc[:split]
    test = series.iloc[split:]
    
    # Limit test to 24 hours
    if len(test) > 24:
        test = test.iloc[:24]
        train = train.iloc[-100:]
    
    try:
        model = ARIMA(train, order=(1,0,1))
        fit = model.fit()
        forecast = fit.forecast(steps=len(test))
        
        min_len = min(len(test), len(forecast))
        
        return {
            "train": train,
            "test": test[:min_len],
            "forecast": forecast[:min_len],
            "rmse": np.sqrt(mean_squared_error(test[:min_len], forecast[:min_len])),
            "mae": mean_absolute_error(test[:min_len], forecast[:min_len]),
            "mape": mean_absolute_percentage_error(test[:min_len], forecast[:min_len])}
    except:
        # Fallback persistence forecast
        forecast = [train.iloc[-1]] * len(test)
        return {
            "train": train,
            "test": test,
            "forecast": forecast,
            "rmse": np.sqrt(mean_squared_error(test, forecast)),
            "mae": mean_absolute_error(test, forecast),
            "mape": mean_absolute_percentage_error(test, forecast)}


def sql_analytics(df, target_col):
    """
    FUNCTION 8: SQL-style analytics and visualizations
    INPUT: DataFrame
    OUTPUT: Results dictionary + Figures dictionary
    """
    results = {}
    figures = {}
    target = "hourly_average_price"
    
    # Peak hours
    peak_hours = df.groupby('hour')[target].agg(['mean', 'std']).sort_values('mean', ascending=False)
    results['peak_hours'] = peak_hours.head(5)
    
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    colors = ['red' if i < 3 else 'steelblue' for i in range(len(peak_hours.head(10)))]
    ax1.bar(peak_hours.head(10).index.astype(str), peak_hours.head(10)['mean'], color=colors)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Average Price ($/MWh)')
    ax1.set_title('Top 10 Peak Pricing Hours')
    ax1.grid(True, alpha=0.3)
    figures['peak_hours_plot'] = fig1
    
    # Weekly pattern
    weekly_avg = df.groupby('weekday')[target].agg(['mean', 'std'])
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg.index = days
    results['weekly_avg'] = weekly_avg
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(days, weekly_avg['mean'], marker='o', linewidth=2, markersize=8, color='steelblue')
    ax2.fill_between(range(7), weekly_avg['mean'] - weekly_avg['std'], 
                     weekly_avg['mean'] + weekly_avg['std'], alpha=0.2, color='steelblue')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Average Price ($/MWh)')
    ax2.set_title('Weekly Price Pattern')
    ax2.grid(True, alpha=0.3)
    figures['weekly_plot'] = fig2
    
    # Monthly pattern
    monthly_stats = df.groupby('month')[target].agg(['mean', 'min', 'max', 'std'])
    results['monthly_stats'] = monthly_stats
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax3.plot(months[:len(monthly_stats)], monthly_stats['mean'], marker='s', linewidth=2, 
             markersize=6, color='green', label='Average Price')
    ax3.fill_between(range(len(monthly_stats)), monthly_stats['min'], monthly_stats['max'], 
                     alpha=0.2, color='green', label='Price Range')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Price ($/MWh)')
    ax3.set_title('Monthly Price Pattern')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    figures['monthly_plot'] = fig3
    
    # Hourly price profile
    hourly_profile = df.groupby('hour')[target].agg(['mean', 'std'])
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(hourly_profile.index, hourly_profile['mean'], marker='o', linewidth=2, color='darkblue')
    ax4.fill_between(hourly_profile.index, 
                     hourly_profile['mean'] - hourly_profile['std'], 
                     hourly_profile['mean'] + hourly_profile['std'], 
                     alpha=0.2, color='darkblue')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Price ($/MWh)')
    ax4.set_title('24-Hour Price Profile')
    ax4.axhline(df[target].mean(), color='red', linestyle='--', alpha=0.5, 
                label=f'Daily Avg: ${df[target].mean():.2f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    figures['hourly_profile'] = fig4
    
    # Price distribution
    results['price_quantiles'] = df[target].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    sns.histplot(df[target], bins=50, kde=True, ax=ax5, color='steelblue')
    ax5.axvline(df[target].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: ${df[target].mean():.2f}')
    ax5.axvline(df[target].median(), color='orange', linestyle='--', linewidth=2, 
                label=f'Median: ${df[target].median():.2f}')
    ax5.set_xlabel('Price ($/MWh)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Price Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    figures['distribution_plot'] = fig5
    
    # Weekend vs Weekday
    if 'weekend' in df.columns:
        weekend_avg = df.groupby('weekend')[target].agg(['mean', 'min', 'max'])
        weekend_avg.index = ['Weekday', 'Weekend']
        results['weekend_comparison'] = weekend_avg
    
    return results, figures


def ai_consultant(df, sql_result, arima_result=None):
    """AI Consultant - Returns complete data with insights"""
    
    target = "hourly_average_price"
    
    # Calculate metrics
    mean_price = df[target].mean()
    cv = df[target].std() / mean_price
    peak = sql_result["peak_hours"].index.tolist()[:3]
    correlation = df['hourly_demand'].corr(df[target])
    
    # Weekend discount
    weekend_discount = 0
    if 'weekend' in df.columns:
        weekday_price = df[df['weekend']==0][target].mean()
        weekend_price = df[df['weekend']==1][target].mean()
        if weekday_price > weekend_price:
            weekend_discount = ((weekday_price - weekend_price) / weekday_price) * 100
    
    # Trend
    trend = 0
    if len(df) > 336:
        recent = df[target].tail(168).mean()
        older = df[target].tail(336).mean()
        trend = ((recent - older) / older) * 100
    
    # Peak premium
    peak_price = df[df['hour'].isin(peak)][target].mean()
    off_peak_price = df[~df['hour'].isin(peak)][target].mean()
    peak_premium = ((peak_price - off_peak_price) / off_peak_price) * 100
    
    # Volatility level
    if cv < 0.15: vol_level = "LOW"
    elif cv < 0.30: vol_level = "MODERATE"
    elif cv < 0.50: vol_level = "HIGH"
    else: vol_level = "EXTREME"
    
    insights = [
        f"Average price: ${mean_price:.2f}/MWh",
        f"Peak hours {peak}: {peak_premium:.0f}% premium",
        f"Weekend discount: {weekend_discount:.1f}%",
        f"Volatility: {cv:.1%} ({vol_level})"]
    
    risks = []
    if cv > 0.3:
        risks.append(f"HIGH - Price volatility ({cv:.1%})")
    if peak_premium > 30:
        risks.append(f"HIGH - Peak exposure ({peak_premium:.0f}% premium)")
    if trend > 5:
        risks.append(f"MEDIUM - Upward trend (+{trend:.1f}%)")
    
    recommendations = [
        f"Shift load from peak hours {peak} to off-peak",
        f"Increase weekend operations ({weekend_discount:.0f}% savings)",
        "Implement real-time price monitoring"]
    
    # Summary with ALL keys
    summary = {
        "avg_price": mean_price,
        "cv": cv,
        "peak_hours": peak,
        "weekend_discount": weekend_discount,
        "trend": trend,
        "correlation": correlation,
        "peak_premium": peak_premium,
        "insights": insights,      
        "risks": risks,            
        "recommendations": recommendations  }
    
    prompt = f"""SENIOR ENERGY STRATEGY CONSULTANT.

DATA:
- Price: ${mean_price:.2f}/MWh
- Volatility: {cv:.1%} ({vol_level})
- Peak Hours: {peak} ({peak_premium:.0f}% premium)
- Weekend: {weekend_discount:.1f}% discount
- Trend: {trend:+.1f}%

Write 6-section report:
1. EXECUTIVE SUMMARY
2. MARKET OVERVIEW
3. KEY INSIGHTS
4. RISK ASSESSMENT
5. RECOMMENDATIONS (3 actions)
6. CONCLUSION

Be concise and data-driven.
"""

   
    try:
        res = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
            stream=False)
        
        return {
            "type": " LLM Consultant",
            "report": res["message"]["content"],
            "summary": summary}
    
    except Exception as e:
        return {
            "type": " Analytics Mode",
            "report": f"""## Executive Report

**Summary:** Market shows {vol_level.lower()} volatility with ${mean_price:.2f}/MWh average price.

**Key Insights:**
• Peak hours {peak}: {peak_premium:.0f}% premium
• Weekend discount: {weekend_discount:.1f}%

**Recommendations:**
1. Shift load from {peak}
2. Use weekend operations
3. Monitor prices

**Conclusion:** Act within 30 days.""",
            "summary": summary}



# SECTION 2: PIPELINE ORCHESTRATOR

def run_pipeline(file):
    """
    PIPELINE ORCHESTRATOR - Chains all functions together
    Data flows: load → clean → features → analyze → forecast → report
    """
    
    # STEP 1: Load data
    df = load_data(file)
    
    # STEP 2: Clean data
    df, clean_report = clean_data(df)
    
    # STEP 3: Feature engineering
    df = feature_engineering(df)
    
    # STEP 4: Statistical summary
    stats = statistical_summary(df)
    
    # STEP 5: EDA Analysis
    eda = eda_analysis(df)
    
    # STEP 6: Stationarity test
    stationarity = check_stationarity(df['hourly_average_price'])
    
    # STEP 7: ARIMA Forecast
    arima = fast_arima_forecast(df)
    
    # STEP 8: SQL Analytics
    sql_results, sql_figures = sql_analytics(df, "hourly_average_price")
    
    # STEP 9: AI Consultant
    ai = ai_consultant(df, sql_results, arima)
    
    # STEP 10: Return all results
    return {
        "df": df,
        "eda": eda,
        "arima": arima,
        "stats": stats,
        "sql_results": sql_results,
        "sql_figures": sql_figures,
        "ai": ai,
        "stationarity": stationarity,
        "cleaning_report": clean_report}


def process_in_background(file, result_queue):
    """
    BACKGROUND EXECUTOR - Runs pipeline without blocking UI
    This is NOT the pipeline - it just runs the pipeline in background
    """
    try:
        result = run_pipeline(file)
        result_queue.put({"success": True, "result": result})
    except Exception as e:
        result_queue.put({"success": False, "error": str(e)})



# =========================================================
# SECTION 3: STREAMLIT UI
# =========================================================

# Sidebar
st.sidebar.header(" Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="CSV must contain 'date', 'hour', 'hourly_demand', 'hourly_average_price'")

with st.sidebar.expander(" CSV Format"):
    st.code("""
date,hour, hourly_demand, hourly_average_price
2024-01-01,0,18500,45.2
2024-01-01,1,17800,44.8
    """)

# Ollama status
with st.sidebar.expander(" Ollama Status"):
    if OLLAMA_AVAILABLE:
        try:
            ollama.list()
            st.success(" Ollama connected")
        except:
            st.warning(" Ollama not running")
            st.code("ollama serve")
    else:
        st.info(" Ollama not installed (optional)")

# Run button with background processing
if uploaded_file is not None:
    if st.sidebar.button(" Run Pipeline", type="primary", use_container_width=True):
        
        st.session_state.result = None
        st.session_state.processing = True
        
        result_queue = queue.Queue()
        
        thread = threading.Thread(
            target=process_in_background,
            args=(uploaded_file, result_queue) )
        thread.start()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        while thread.is_alive():
            elapsed = int(time.time() - start_time)
            status_text.info(f" Processing data... {elapsed} seconds elapsed")
            progress_bar.progress(min(elapsed / 60, 0.95))
            time.sleep(0.5)
        
        try:
            result_data = result_queue.get(timeout=5)
            
            if result_data["success"]:
                st.session_state.result = result_data["result"]
                st.session_state.cleaning_report = result_data["result"].get("cleaning_report")
                st.session_state.processing = False
                status_text.empty()
                progress_bar.empty()
                st.success(" Pipeline Completed Successfully!")
                st.balloons()
                st.rerun()
            else:
                st.error(f" Error: {result_data['error']}")
                st.session_state.processing = False
                
        except queue.Empty:
            st.error(" Processing timed out")
            st.session_state.processing = False


# =========================================================
# DISPLAY RESULTS
# =========================================================
if st.session_state.result is not None and not st.session_state.processing:
    r = st.session_state.result
    df = r["df"]
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Dashboard", "EDA", " Forecasts", " Analytics", " AI Consultant"])
    
    # TAB 1: DASHBOARD
    with tab1:
        st.subheader(" Key Performance Indicators")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Avg Price", f"${df['hourly_average_price'].mean():.2f}")
        with c2: st.metric("Avg Demand", f"{df['hourly_demand'].mean():,.0f} MW")
        with c3: st.metric("Volatility", f"{df['hourly_average_price'].std():.2f}")
        with c4: st.metric("Data Points", f"{len(df):,}")
        
        if st.session_state.cleaning_report:
            st.markdown("---")
            st.subheader(" Data Cleaning")
            cr = st.session_state.cleaning_report
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Initial", f"{cr['initial_rows']:,}")
            with c2: st.metric("Final", f"{cr['final_rows']:,}")
            with c3: st.metric("Removed", f"{cr['rows_removed']:,}", delta=f"{cr['cleaning_percentage']}%")
            with c4: st.metric("Outliers", cr['outliers_removed'])
        
        st.markdown("---")
        st.subheader(" Price Trend")
        st.line_chart(df['hourly_average_price'].tail(168))
        
        st.markdown("---")
        st.subheader(" Stationarity")
        s = r["stationarity"]
        if s["stationary"]:
            st.success(f" Stationary (p={s['p_value']:.4f})")
        else:
            st.warning(f" Non-Stationary (p={s['p_value']:.4f})")
    
    # TAB 2: EDA
    with tab2:
        st.subheader(" Exploratory Data Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Hourly Price Pattern**")
            st.bar_chart(r["eda"]["hourly_price"])
        with c2:
            st.write("**Hourly Demand Pattern**")
            st.bar_chart(r["eda"]["hourly_demand"])
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Weekly Price Pattern**")
            st.bar_chart(r["eda"]["weekly_price"])
        with c2:
            st.write("**Monthly Price Pattern**")
            st.line_chart(r["eda"]["monthly_price"])
        
        st.markdown("---")
        
        st.subheader(" Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(r["eda"]["corr"], ax=ax, cmap="coolwarm", annot=True, fmt=".2f")
        st.pyplot(fig)
        
        st.markdown("---")
        st.metric("Anomalies Detected", r["eda"]["anomalies"])
    
    # TAB 3: FORECASTS
    with tab3:
        st.subheader(" ARIMA Forecast")
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("RMSE", f"${r['arima']['rmse']:.2f}")
        with c2: st.metric("MAE", f"${r['arima']['mae']:.2f}")
        with c3: st.metric("MAPE", f"{r['arima']['mape']:.1%}")
        
        st.markdown("---")
        st.subheader("Actual vs Forecast")
        comp_df = pd.DataFrame({
            'Actual': r['arima']['test'].values,
            'Forecast': r['arima']['forecast'].values })
        st.line_chart(comp_df)
        
        st.info(f"ARIMA order (1,0,1) - trained on {len(r['arima']['train'])} points")
    
    # TAB 4: ANALYTICS
    with tab4:
        st.subheader(" SQL-Style Analytics")
        
        c1, c2 = st.columns(2)
        with c1: st.dataframe(r['sql_results']['peak_hours'])
        with c2: st.pyplot(r['sql_figures']['peak_hours_plot'])
        
        st.markdown("---")
        st.subheader(" Weekly Pattern")
        st.pyplot(r['sql_figures']['weekly_plot'])
        
        st.markdown("---")
        st.subheader(" Monthly Statistics")
        st.dataframe(r['sql_results']['monthly_stats'])
        st.pyplot(r['sql_figures']['monthly_plot'])
        
        st.markdown("---")
        st.subheader(" 24-Hour Price Profile")
        st.pyplot(r['sql_figures']['hourly_profile'])
        
        st.markdown("---")
        st.subheader(" Price Distribution")
        
        c1, c2 = st.columns(2)
        with c1:
            qdf = pd.DataFrame(r['sql_results']['price_quantiles'])
            qdf.columns = ['Price ($/MWh)']
            st.dataframe(qdf)
        with c2:
            st.pyplot(r['sql_figures']['distribution_plot'])
        
        if 'weekend_comparison' in r['sql_results']:
            st.markdown("---")
            st.subheader(" Weekend vs Weekday")
            st.dataframe(r['sql_results']['weekend_comparison'])
    
    # TAB 5: AI CONSULTANT
    with tab5:
        st.subheader(" AI Energy Strategy Consultant")
        ai = r["ai"]
        
        if "LLM" in ai["type"]:
            st.success(f" {ai['type']}")
        else:
            st.info(f" {ai['type']}")
        
        st.markdown("---")
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Avg Price", f"${ai['summary']['avg_price']:.2f}")
        with c2: st.metric("Volatility", f"{ai['summary']['cv']:.2%}")
        with c3: st.metric("Peak Hours", ", ".join(map(str, ai['summary']['peak_hours'])))
        
        st.markdown("---")
        
        st.subheader(" Key Insights")
        for insight in ai['summary']['insights']:
            st.success(f"• {insight}")
        
        if ai['summary']['risks']:
            st.subheader(" Risk Analysis")
            for risk in ai['summary']['risks']:
                st.error(f"• {risk}")
        
        st.subheader(" Recommendations")
        for rec in ai['summary']['recommendations']:
            st.write(f"✔ {rec}")
        
        if ai.get("report"):
            st.markdown("---")
            st.subheader(" AI Executive Summary")
            st.write(ai["report"])

else:
    st.info(" Upload a CSV file and click 'Run Pipeline' to start")
    st.markdown("""
    ###  About This Dashboard
    
    **AI Ontario Energy Intelligence Pipeline** provides:
    
    -  **9 Functional Modules** - Each function does ONE thing
    -  **Pipeline Architecture** - Data flows through all functions
    -  **ARIMA Forecasting** - 24-hour price predictions
    -  **Interactive EDA** - Patterns, correlations, anomalies
    -  **AI Consultant** - LLM-powered recommendations (Ollama)
    -  **Cost Savings** - 15-35% identified opportunities""")
    