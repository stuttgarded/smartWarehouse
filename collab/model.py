import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data_ready.csv')
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data.dropna(subset=['date'], inplace=True)
data = data.set_index('date')

# Split periods
start_date_1 = '2024-02-29'
end_date_1 = '2024-06-29'
start_date_2 = '2025-02-28'
end_date_2 = '2025-05-21'

data_period_1 = data.loc[start_date_1:end_date_1]
data_period_2 = data.loc[start_date_2:end_date_2]

print(f"Training data shape: {data_period_1.shape}")
print(f"Training data period: {data_period_1.index[0]} to {data_period_1.index[-1]}")

# Define features and targets
# Features: 12 kolom terakhir (weather + cafe values)
# Targets: semua kolom produk (kecuali 12 kolom terakhir)
x = data_period_1.iloc[:, -12:].values  # Weather features
y = data_period_1.iloc[:, :-12].values  # Product sales

print(f"Features shape: {x.shape}")
print(f"Targets shape: {y.shape}")

# Get product names for later use
product_names = data_period_1.columns[:-12].tolist()
print(f"Products to predict: {len(product_names)} items")

# Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

# Create time series dataset
def create_dataset(x, y, time_steps=7):
    Xs, Ys = [], []
    for i in range(len(x) - time_steps):
        Xs.append(x[i:(i + time_steps)])
        Ys.append(y[i + time_steps])
    return np.array(Xs), np.array(Ys)

time_steps = 7
X_lstm, Y_lstm = create_dataset(x_scaled, y_scaled, time_steps)

print(f"LSTM input shape: {X_lstm.shape}")
print(f"LSTM output shape: {Y_lstm.shape}")

# Split train/validation
split_ratio = 0.8
split_idx = int(len(X_lstm) * split_ratio)

X_train, X_val = X_lstm[:split_idx], X_lstm[split_idx:]
Y_train, Y_val = Y_lstm[:split_idx], Y_lstm[split_idx:]

# Build LSTM model
model = Sequential([
    Input(shape=(X_lstm.shape[1], X_lstm.shape[2])),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(Y_lstm.shape[1])
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Training
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=16,
    verbose=1
)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
y_val_pred_inv = scaler_y.inverse_transform(y_val_pred)
y_val_true_inv = scaler_y.inverse_transform(Y_val)

mse = mean_squared_error(y_val_true_inv, y_val_pred_inv)
mae = mean_absolute_error(y_val_true_inv, y_val_pred_inv)
print(f"\nValidation MSE: {mse:.2f}")
print(f"Validation MAE: {mae:.2f}")

# ==========================================
# PREDIKSI 1 MINGGU KE DEPAN
# ==========================================

def predict_next_week(model, last_sequence_x, scaler_x, scaler_y, days=7):
    """
    Predict next 'days' using the last sequence of features
    """
    predictions = []
    current_sequence = last_sequence_x.copy()
    
    for day in range(days):
        # Predict next day
        next_pred_scaled = model.predict(current_sequence.reshape(1, time_steps, -1), verbose=0)
        predictions.append(next_pred_scaled[0])
        
        # For multi-step prediction, we need next day's features
        # Since we don't have future weather data, we'll use the last available features
        # In real scenario, you would need weather forecast data
        next_features = current_sequence[-1].copy()  # Use last day's features
        
        # Update sequence for next prediction
        current_sequence = np.vstack([current_sequence[1:], next_features])
    
    return np.array(predictions)

# Get the last sequence for prediction
last_sequence_x = x_scaled[-time_steps:]

# Predict next 7 days
next_week_pred_scaled = predict_next_week(model, last_sequence_x, scaler_x, scaler_y, days=7)

# Inverse transform predictions
next_week_pred = scaler_y.inverse_transform(next_week_pred_scaled)

# Create dates for next week
last_date = data_period_1.index[-1]
next_week_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')

# Create prediction DataFrame
prediction_df = pd.DataFrame(
    next_week_pred,
    index=next_week_dates,
    columns=product_names
)

print("\n" + "="*60)
print("PREDIKSI DEMAND 1 MINGGU KE DEPAN")
print("="*60)
print(prediction_df.round(0))

# Show top 10 products for each day
print("\n" + "="*60)
print("TOP 10 PRODUK SETIAP HARI")
print("="*60)

for date in next_week_dates:
    print(f"\n{date.strftime('%Y-%m-%d')} ({date.strftime('%A')}):")
    day_demand = prediction_df.loc[date].sort_values(ascending=False)
    top_10 = day_demand.head(10)
    for i, (product, demand) in enumerate(top_10.items(), 1):
        print(f"  {i:2d}. {product}: {demand:.0f}")

# Total demand per product for the week
weekly_total = prediction_df.sum().sort_values(ascending=False)
print(f"\n" + "="*60)
print("TOTAL DEMAND MINGGUAN (TOP 15)")
print("="*60)
for i, (product, total) in enumerate(weekly_total.head(15).items(), 1):
    print(f"{i:2d}. {product}: {total:.0f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Sales Forecasting Results', fontsize=16)

# 1. Training history
axes[0,0].plot(history.history['loss'], label='Training Loss')
axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
axes[0,0].set_title('Model Training Loss')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('MSE Loss')
axes[0,0].legend()
axes[0,0].grid(True)

# 2. Actual vs Predicted (sample product)
sample_product_idx = 0
axes[0,1].plot(y_val_true_inv[:, sample_product_idx], label='Actual', alpha=0.7)
axes[0,1].plot(y_val_pred_inv[:, sample_product_idx], label='Predicted', alpha=0.7)
axes[0,1].set_title(f'Validation: {product_names[sample_product_idx]}')
axes[0,1].set_xlabel('Days')
axes[0,1].set_ylabel('Sales')
axes[0,1].legend()
axes[0,1].grid(True)

# 3. Next week predictions - total daily demand
daily_totals = prediction_df.sum(axis=1)
axes[1,0].bar(range(7), daily_totals.values)
axes[1,0].set_title('Total Daily Demand - Next Week')
axes[1,0].set_xlabel('Day')
axes[1,0].set_ylabel('Total Demand')
axes[1,0].set_xticks(range(7))
axes[1,0].set_xticklabels([d.strftime('%a') for d in next_week_dates], rotation=45)
axes[1,0].grid(True, alpha=0.3)

# 4. Top products next week
top_products = weekly_total.head(10)
axes[1,1].barh(range(len(top_products)), top_products.values)
axes[1,1].set_title('Top 10 Products - Weekly Demand')
axes[1,1].set_xlabel('Total Demand')
axes[1,1].set_yticks(range(len(top_products)))
axes[1,1].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                          for name in top_products.index], fontsize=8)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save predictions to CSV
prediction_df.to_csv('sales_forecast_next_week.csv')
print(f"\nPrediksi disimpan ke 'sales_forecast_next_week.csv'")

# Model evaluation summary
print(f"\n" + "="*60)
print("MODEL EVALUATION SUMMARY")
print("="*60)
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Number of products: {len(product_names)}")
print(f"Number of features: {x.shape[1]}")
print(f"Time steps: {time_steps}")
print(f"Final Validation MSE: {mse:.2f}")
print(f"Final Validation MAE: {mae:.2f}")