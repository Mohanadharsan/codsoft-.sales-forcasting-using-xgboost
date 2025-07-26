import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Sales Forecasting App", layout="wide")
st.title(" Sales Forecasting with XGBoost")
uploaded_file = st.file_uploader("multi_item_sales_with_features.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(" File uploaded successfully!")
else:
    st.info("Using default dataset: multi_item_sales_with_features.csv")
    df = pd.read_csv("multi_item_sales_with_features.csv")
df['date'] = pd.to_datetime(df['date'])
if 'prev_sales' not in df.columns or 'dayofweek' not in df.columns or 'weekofyear' not in df.columns:
    df['prev_sales'] = df.groupby('item')['sales'].shift(1)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

df.dropna(inplace=True)
items = df['item'].unique()
selected_item = st.sidebar.selectbox("Select Item:", items)
item_df = df[df['item'] == selected_item].copy()
item_df = item_df[item_df['prev_sales'] > 0]

features = ['prev_sales', 'dayofweek', 'weekofyear']
target = 'sales'
X = item_df[features]
y = item_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.metric("Mean Absolute Error", f"{mae:.2f}")
plot_df = item_df.loc[y_test.index].copy()
plot_df['Predicted'] = y_pred
st.subheader(f" Actual vs Predicted Sales for {selected_item}")
st.line_chart(plot_df.set_index('date')[['sales', 'Predicted']])

# ---- Forecast Tomorrow ----
if st.button(" Forecast Tomorrow's Sales"):
    last_row = item_df.iloc[-1]
    future = pd.DataFrame({
        'prev_sales': [last_row['sales']],
        'dayofweek': [(last_row['date'] + pd.Timedelta(days=1)).dayofweek],
        'weekofyear': [(last_row['date'] + pd.Timedelta(days=1)).isocalendar().week]
    })
    forecast = model.predict(future)[0]
    st.success(f"Tomorrow's forecast for {selected_item}: {forecast:.2f}")