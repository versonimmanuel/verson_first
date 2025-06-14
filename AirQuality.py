import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
df = pd.read_csv("Copy of TamilNadu.csv")
df["From Date"] = pd.to_datetime(df["From Date"], errors='coerce')
df = df.dropna(subset=["From Date"])
df = df.sort_values("From Date")
daily_pm25 = df.groupby("From Date")["PM2.5"].mean().reset_index()
daily_pm25.columns = ["Date", "PM2.5"]
daily_pm25['day'] = daily_pm25['Date'].dt.day
daily_pm25['month'] = daily_pm25['Date'].dt.month
daily_pm25['year'] = daily_pm25['Date'].dt.year
daily_pm25['dayofweek'] = daily_pm25['Date'].dt.dayofweek
X = daily_pm25[['day', 'month', 'year', 'dayofweek']]
y = daily_pm25['PM2.5']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Evaluation Metrics:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}")
num_days = int(input("Enter number of future days to predict PM2.5: "))
last_actual_date = daily_pm25["Date"].max()
today = pd.to_datetime(datetime.today().date())
end_pred_date = today + timedelta(days=num_days)
start_pred_date = last_actual_date + timedelta(days=1)
all_future_dates = pd.date_range(start=start_pred_date, end=end_pred_date)
future_df = pd.DataFrame({'Date': all_future_dates})
future_df['day'] = future_df['Date'].dt.day
future_df['month'] = future_df['Date'].dt.month
future_df['year'] = future_df['Date'].dt.year
future_df['dayofweek'] = future_df['Date'].dt.dayofweek
future_df['Predicted PM2.5'] = model.predict(future_df[['day', 'month', 'year','dayofweek']])
def get_status(pm25):
    if pm25 <= 30:
        return "Good"
    elif pm25 <= 60:
        return "Moderate"
    elif pm25 <= 90:
        return "Unhealthy for Sensitive Groups"
    elif pm25 <= 150:
        return "Unhealthy"
    elif pm25 <= 250:
        return "Very Unhealthy"
    else:
        return "Hazardous"
future_df['Status'] = future_df['Predicted PM2.5'].apply(get_status)
next_days = future_df[future_df['Date'] > today].head(num_days)
plt.figure(figsize=(14, 6))
plt.plot(daily_pm25['Date'], daily_pm25['PM2.5'], color='green', linestyle='--',
linewidth=1, marker='.', label='Actual PM2.5')
plt.plot(future_df['Date'], future_df['Predicted PM2.5'], color='blue', linestyle='--',linewidth=1, marker='.', label='Predicted PM2.5')
plt.title("PM2.5: Actual vs Predicted pm 2.5 value for Tamil Nadu")
plt.xlabel("Date")
plt.ylabel("PM2.5 Level of Tamil Nadu")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
importances = model.feature_importances_
feature_names = ['day', 'month', 'year', 'dayofweek']
plt.figure(figsize=(7,5))
plt.barh(feature_names, importances, color='skyblue')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,5))
plt.hist(daily_pm25['PM2.5'], bins=20, color='orange', edgecolor='black')
plt.title("Distribution of PM2.5 Levels (Historical Data)")
plt.xlabel("PM2.5 Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
status_counts = next_days['Status'].value_counts()
status_counts.plot(kind='bar', color='purple', edgecolor='black')
plt.title(f"TAMIL NADU Air Quality Status Forecast for Next {num_days} Days")
plt.ylabel("Number of Days")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
heatmap_data = daily_pm25.pivot_table(values='PM2.5', index='dayofweek', columns='month', aggfunc='mean')
plt.figure(figsize=(10,6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='coolwarm')
plt.title("Average PM2.5 by Weeks and Days in Tamil Nadu")
plt.xlabel("Month")
plt.ylabel("Day of Week (0=Mon, 6=Sun)")
plt.tight_layout()
plt.show()
print(f"\n Predicted PM2.5 for the Next {num_days} Days in Tamil Nadu:\n")
print(next_days[['Date', 'Predicted PM2.5', 'Status']].to_string(index=False))
