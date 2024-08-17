import numpy as np
import plotly.express as px
import pandas as pd

def interactive_plot(df):
    fig = px.line()  # Initialize an empty figure object
    
    # Adding scatter plots for each column except the 'Date' column
    for col in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[col], mode='lines', name=col)
    
    fig.update_layout(
        width=450,
        margin=dict(l=20, r=20, t=10, b=20),
        yaxis_title='Price'
    )
    return fig

# Function to normalize the prices based on the initial price
# Function to normalize the prices based on the initial price

def normalize(df):
    df_normalized = df.copy()
    for col in df.columns[1:]:
        df_normalized[col] = df[col] / df[col].iloc[0]  # Normalize based on the initial price
    return df_normalized

# Function to calculate daily returns
def daily_returns(df):
    df_daily_return = df.copy()
    for col in df.columns[1:]:
        df_daily_return[col] = df[col].pct_change() * 100  # Calculate daily percentage change
    df_daily_return['Date'] = df['Date']  # Ensure the 'Date' column is preserved
    return df_daily_return 

 #function to calculate beta
 

def calculate_beta(stocks_daily_return, stock):
    # Check if 'SP500' column exists
    if 'SP500' not in stocks_daily_return.columns:
        raise KeyError("'SP500' column not found in DataFrame")
    
    # Check if the stock column exists
    if stock not in stocks_daily_return.columns:
        raise KeyError(f"'{stock}' column not found in DataFrame")

    # Calculate daily returns for the stock and S&P 500
    stock_returns = stocks_daily_return[stock]
    sp500_returns = stocks_daily_return['SP500']

    
    # Ensure no NaN values for regression
    valid_data = stock_returns.notna() & sp500_returns.notna()
    stock_returns = stock_returns[valid_data]
    sp500_returns = sp500_returns[valid_data]

    # Perform linear regression to find beta
    beta, alpha = np.polyfit(sp500_returns, stock_returns, 1)
    
    return beta, alpha

