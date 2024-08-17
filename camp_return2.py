import numpy as np
import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime
import plotly.express as px

# Import the interactive_plot function from camp_functions
from camp_functions import calculate_beta, daily_returns, interactive_plot, normalize

st.set_page_config(page_title="CAPM",
                   page_icon="chart_with_upwards_trend",
                   layout='wide')

st.title("Capital Asset Pricing Model")

# Getting user input
col1, col2 = st.columns([1, 1])
with col1:
    stocks_list = st.multiselect("Choose 4 stocks", 
                                 ['TSLA', 'AAPL', 'NFLX', 'MGM', 'AMZN', 'NVDA', 'GOOGL', 'SAIPEM'],
                                 ['TSLA', 'AAPL', 'AMZN', 'GOOGL'])
with col2:
    years = st.number_input("Number of years", 1, 10)


# Downloading data for S&P 500
end = datetime.today()
start = datetime(end.year - years, end.month, end.day)

try:
    SP500 = pdr.get_data_fred('SP500', start, end)
    st.write("S&P 500 Data:")
    st.write(SP500.head())  # Display the data in the Streamlit app
except Exception as e:
    st.error(f"An error occurred while fetching S&P 500 data: {e}")

# Downloading and processing stock data
stocks_df = pd.DataFrame()
for stock in stocks_list:
    try:
        data = yf.download(stock, start=start, end=end)
        stocks_df[stock] = data['Close']
    except Exception as e:
        st.error(f"An error occurred while fetching data for {stock}: {e}")

# Reset indices for merging
stocks_df.reset_index(inplace=True)
SP500.reset_index(inplace=True)
SP500.columns = ['Date', 'SP500']

# Ensure 'Date' columns are in datetime format
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
SP500['Date'] = pd.to_datetime(SP500['Date'])

# Merge dataframes on 'Date'
combined_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

# Display the dataframes in Streamlit
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### Dataframe Head")
    st.dataframe(combined_df.head(), use_container_width=True)
with col2:
    st.markdown("### Dataframe Tail")
    st.dataframe(combined_df.tail(), use_container_width=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### Price of All Stocks")
    st.plotly_chart(interactive_plot(stocks_df))
with col2:
    # Apply normalization
    normalized_df = normalize(stocks_df)
    st.markdown("### Price of All Stocks - Normalized")
    st.plotly_chart(interactive_plot(normalized_df))
    
    stocks_daily_return = daily_returns(stocks_df)
    st.markdown("### Daily Returns of All Stocks")
    st.dataframe(stocks_daily_return.head(), use_container_width=True)


beta = {}
alpha = {}

# Calculate beta and alpha for each stock
# Print column names to debug
st.write("Columns in stocks_daily_return DataFrame:", stocks_daily_return.columns.tolist())

# Initialize dictionaries to store beta and alpha values
beta = {}
alpha = {}

# Calculate beta and alpha for each stock
for i in stocks_daily_return.columns:
    if i != 'Date' and i != 'sp500':
        try:
            b, a = calculate_beta(stocks_daily_return, i)
            beta[i] = b
            alpha[i] = a
        except KeyError as e:
            st.error(f"KeyError: {e}")

# Print beta and alpha values for debugging
st.write("Beta values:", beta)
st.write("Alpha values:", alpha)

# Create DataFrame to display beta values
beta_df = pd.DataFrame(columns=['Stock', 'Beta Value'])
beta_df['Stock'] = list(beta.keys())
beta_df['Beta Value'] = [round(b, 2) for b in beta.values()]

# Display the beta DataFrame in Streamlit
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('### Calculated Beta Values')
    st.dataframe(beta_df, use_container_width=True)

# Calculate the risk-free rate and market return
rf = 0  # Risk-free rate (adjust as needed)
rm = stocks_daily_return['sp500'].mean() * 252  # Expected market return (annualized)

# Calculate returns using the CAPM model
return_value = []
for stock, value in beta.items():
    capm_return = rf + (value * (rm - rf))
    return_value.append(round(capm_return, 2))

# Create DataFrame for CAPM returns
return_df = pd.DataFrame()
return_df['Stock'] = list(beta.keys())
return_df['Return Value'] = return_value

# Display the CAPM returns DataFrame in Streamlit
with col2:
    st.markdown('### Calculated Return Using CAPM')
    st.dataframe(return_df, use_container_width=True)

