import yfinance as yf

print("Downloading INTC data...")
# Download 7 days of 1-minute data for Intel
data = yf.download("INTC", period="7d", interval="1m")

# Save it to a CSV file in the current folder
data.to_csv("INTC_1min_data.csv")
print("Success! Saved as INTC_1min_data.csv")