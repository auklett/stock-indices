# %%
import pandas as pd
import yfinance as yf
import xgboost as xgb
import matplotlib.pyplot as plt

# %%
last_n_days = 400
indices = ["^GSPC", "^DJI", "^IXIC", "^RUT", "^FTSE"]
index_names = ["S&P 500", "Dow Jones", "NASDAQ Composite", "Russell 2000", "FTSE 100"]


# %%
index_acc = []
for i, ndx in enumerate(indices):
    stockndx = yf.Ticker(ndx)
    stockndx = stockndx.history(period="max")

    #
    del stockndx["Dividends"]
    del stockndx["Stock Splits"]
    stockndx["After7"] = stockndx["Close"].shift(-7)
    stockndx = stockndx[:-7]

    #
    train = stockndx.iloc[:-last_n_days]
    test = stockndx.iloc[-last_n_days:]

    # Get rid of data before year
    stockndx = stockndx.loc["2000-01-01":].copy()
    features = ["Close", "Volume", "Open", "High", "Low"]

    #  
    model7 = xgb.XGBRegressor(booster="dart", 
                              seed=42, 
                              eval_metric="rmsle", 
                              n_estimators=100, 
                              learning_rate=0.1)
    model7.fit(train[features], train["After7"], verbose=True)
       
    preds7 = model7.predict(test[features])
    
    accuracy7 = model7.score(test[features], test["After7"])
    index_acc.append(accuracy7)
    print(index_names[i])
    print(accuracy7)

    # Plot Predictions vs Actual
    plt.plot(stockndx["Close"], label="Close Price")
    plt.plot(test["After7"].index, preds7, label="Predictions")
    plt.title(index_names[i] + " (2000-2024)")
    plt.xlabel("Year")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()
 
    plt.plot(stockndx["Close"][-last_n_days:], label="Close Price")
    plt.plot(test["After7"].index, preds7, label="Predictions")
    plt.xticks(rotation = 36)
    plt.title(index_names[i] + " (2022-2024)")
    plt.xlabel("Year-Month")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

# %%
for i in range(len(index_acc)):
    print(index_acc[i], "\t", index_names[i])
# %%

"""
0.7432105069122489 	     S&P 500
0.5945033469960961 	     Dow Jones
0.9123815716109644 	     NASDAQ Composite
0.45400785050111736 	 Russell 2000
0.4584177496540084 	     FTSE 100
"""
    