import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')


df = pd.read_csv("multi_item_sales_with_features.csv")


df['date'] = pd.to_datetime(df['date'])

if 'prev_sales' not in df.columns or 'dayofweek' not in df.columns or 'weekofyear' not in df.columns:
    df['prev_sales'] = df.groupby('item')['sales'].shift(1)  
    df['dayofweek'] = df['date'].dt.dayofweek               
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)


df.dropna(inplace=True)
all_predictions = []
mae_scores = []
items = df['item'].unique()

for item in items:
    item_df = df[df['item'] == item].copy()

  
    features = ['prev_sales', 'dayofweek', 'weekofyear']
    target = 'sales'

    item_df = item_df[item_df['prev_sales'] > 0]
    X = item_df[features]
    y = item_df[target]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append((item, mae))

    
    item_results = pd.DataFrame({
        'date': item_df.loc[y_test.index, 'date'],
        'actual': y_test.values,
        'predicted': y_pred,
        'item': item
    })
    all_predictions.append(item_results)
final_predictions = pd.concat(all_predictions)
print(final_predictions.head())
print("\n Mean Absolute Error per item:")
for item, score in mae_scores:
  print(f"{item}: {score:.2f}")
item_to_plot = 'item_1'  
plot_df = final_predictions[final_predictions['item'] == item_to_plot]
plt.figure(figsize=(12, 5))
plt.plot(plot_df['date'], plot_df['actual'], label="Actual")
plt.plot(plot_df['date'], plot_df['predicted'], label="Predicted")
plt.title(f"Sales Forecast for {item_to_plot}")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()