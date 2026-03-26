import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def fetch_data(ticker="^GSPC", start_date="2021-01-01", end_date="2025-12-31"):
    """
    獲取 S&P 500 歷史資料
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    # 設定 auto_adjust=False 確保與舊版 yfinance 行為一致，此處為單純取得 Close
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def feature_engineering(df):
    """
    資料預處理與特徵工程，加入技術指標 (MA, RSI, MACD)，並產生目標變數
    """
    print("Performing feature engineering...")
    
    # 處理 yfinance 可能回傳的多層 MultiIndex (Column level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 確保資料為按照日期排序 (防呆處理)
    df.sort_index(inplace=True)
    
    # 1. 基本技術指標 (Moving Averages)
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # 2. RSI (相對強弱指標) - 14天期
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. 目標變數設定：次日的實際收盤價 (Next Day's Close Price)
    # 如此模型才能在當天學習到明天的價格
    df['Target_Next_Close'] = df['Close'].shift(-1)
    
    # 5. 清理因指標落後天數以及 shift(-1) 產生的缺失值 (NaN)
    df.dropna(inplace=True)
    
    return df

def main():
    # 1. 資料獲取
    df = fetch_data()
    if df.empty:
        print("未下載到任何資料，請檢查網路連線或 yfinance 的狀態。")
        return
        
    # 2. 特徵工程
    df = feature_engineering(df)
    
    # 3. 嚴格依照時間順序切割資料
    # 禁止使用 random shuffle 以避免未來函數 (look-ahead bias)
    # 訓練集：2021-01-01 至 2024-12-31
    # 測試集：2025-01-01 至 2025-12-31
    train_mask = (df.index >= '2021-01-01') & (df.index <= '2024-12-31')
    test_mask = (df.index >= '2025-01-01') & (df.index <= '2025-12-31')
    
    train_data = df.loc[train_mask]
    test_data = df.loc[test_mask]
    
    print(f"Training data shape (2021-2024): {train_data.shape}")
    print(f"Testing data shape (2025): {test_data.shape}")
    
    if test_data.empty:
        print("警告：測試集為空，可能是因為這段期間內沒有資料！")
        return
    
    # 選擇投入模型的特徵與目標變數
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_20', 'RSI', 'MACD', 'MACD_Signal']
    target = 'Target_Next_Close'
    
    X_train = train_data[features]
    y_train = train_data[target]
    
    X_test = test_data[features]
    y_test = test_data[target]
    
    # 4. 模型訓練
    print("\nTraining Random Forest model...")
    # 設定隨機森林的基礎參數並訓練
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    print("Training XGBoost model...")
    # 設定 XGBoost 的基礎參數並訓練
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    # 5. 模型驗證與評估指標計算
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)
    
    # 計算均方誤差 (MSE) 越小越好
    rf_mse = mean_squared_error(y_test, rf_preds)
    xgb_mse = mean_squared_error(y_test, xgb_preds)
    
    print("\n" + "="*45)
    print("模型評估結果 (MSE on 2025 Test Set):")
    print(f"Random Forest MSE : {rf_mse:.2f}")
    print(f"XGBoost MSE       : {xgb_mse:.2f}")
    print("="*45 + "\n")
    
    # 6. 結果視覺化 (繪製 2025 年的價格比較圖)
    print("Plotting actual vs predicted prices...")
    
    # 設定圖表長寬比例
    plt.figure(figsize=(14, 7))
    
    # 畫三條折線：實際股價、隨機森林預測、XGBoost預測
    plt.plot(test_data.index, y_test, label='Actual Price (Next Day Close)', color='black', linewidth=2)
    plt.plot(test_data.index, rf_preds, label='Random Forest Prediction', color='blue', linestyle='--', alpha=0.8)
    plt.plot(test_data.index, xgb_preds, label='XGBoost Prediction', color='red', linestyle='-.', alpha=0.8)
    
    # 優化圖表外觀
    plt.title('S&P 500 Price Prediction for 2025 (Test Set)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='best', fontsize=12) # 加入圖例
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # 將圖表存成圖片檔案
    plot_filename = 'sp500_prediction_2025.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot successfully saved as '{plot_filename}'")
    
    # 若在 Notebook 情境下，可自動顯示 (但在腳本執行時不中斷程式)
    # plt.show()

if __name__ == "__main__":
    main()
