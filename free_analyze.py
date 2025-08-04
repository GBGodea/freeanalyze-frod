import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

COLUMNS = [
    'transaction_id', 'customer_id', 'timestamp', 'amount', 'currency',
    'country', 'city', 'device', 'ip_address', 'vendor_category',
    'is_fraud', 'is_high_risk_vendor', 'is_outside_home_country', 'is_weekend',
    'last_hour_activity'
]
CATEGORICAL_COLS = ['currency', 'country', 'city', 'device', 'vendor_category']

TRANSACTION_PATH = 'data/transaction_fraud_data.parquet'
transactions = pd.read_parquet(
    TRANSACTION_PATH,
    columns=COLUMNS,
    engine='pyarrow'
)

transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], utc=True).dt.tz_convert(None)
for col in CATEGORICAL_COLS:
    transactions[col] = transactions[col].astype('category')
bools = ['is_fraud', 'is_high_risk_vendor', 'is_outside_home_country', 'is_weekend']
for col in bools:
    transactions[col] = transactions[col].astype('bool')

transactions['hour'] = transactions['timestamp'].dt.hour.astype('int8')
transactions['day_of_week'] = transactions['timestamp'].dt.dayofweek.astype('int8')
transactions['date'] = transactions['timestamp'].dt.normalize()

exchange = pd.read_parquet('data/historical_currency_exchange.parquet', engine='pyarrow')
exchange['date'] = pd.to_datetime(exchange['date'])
exchange_long = (
    exchange
    .melt(id_vars=['date'], var_name='currency', value_name='rate_to_usd')
    .astype({'currency':'category'})
)
transactions = transactions.merge(
    exchange_long,
    how='left',
    on=['date', 'currency'],
    validate='m:1'
)
transactions['amount_usd'] = (transactions['amount'] / transactions['rate_to_usd']).astype('float32')

del exchange, exchange_long
gc.collect()

mem_mb = transactions.memory_usage(deep=True).sum() / 1024**2
print(f"Память: {mem_mb:.2f} MB")
print(f"Уровень мошенничества: {transactions['is_fraud'].mean():.2%}")

sample = transactions.sample(frac=0.1, random_state=42)
# Распределение сумм
plt.figure(figsize=(10, 6))
sns.histplot(
    data=sample,
    x='amount_usd',
    hue='is_fraud',
    bins=50,
    log_scale=True,
    element='step',
    stat='density'
)
plt.title('Распределение суммы транзакций в USD (лог-шкала)')
plt.xlabel('USD')
plt.ylabel('Плотность')
plt.tight_layout()
plt.show()

# Корреляция
numeric_cols = ['amount_usd', 'hour', 'day_of_week', 'is_fraud']
corr = transactions[numeric_cols].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, fmt='.2f', square=True)
plt.title('Корреляция признаков')
plt.tight_layout()
plt.show()

# Фрод ip-адреса
fraud_ips = (
    transactions.loc[transactions['is_fraud'], 'ip_address']
    .value_counts()
    .nlargest(10)
    .to_frame(name='count')
    .reset_index()
    .rename(columns={'index': 'ip_address'})
)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=fraud_ips,
    x='count',
    y='ip_address'
)
plt.title('Топ 10 IP-адресов с наибольшим числом мошеннических транзакций')
plt.xlabel('Количество фрод-транзакций')
plt.ylabel('IP-адрес')
plt.tight_layout()
plt.show()

# фрод девайсы
fraud_devices = (
    transactions.loc[transactions['is_fraud'], 'device']
    .value_counts()
    .nlargest(10)
    .to_frame(name='count')
    .reset_index()
    .rename(columns={'index': 'device'})
)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=fraud_devices,
    x='count',
    y='device'
)
plt.title('Топ 10 устройств с наибольшим числом мошеннических транзакций')
plt.xlabel('Количество фрод-транзакций')
plt.ylabel('Устройство')
plt.tight_layout()
plt.show()

# Самое активное время суток по активности мошенников
hourly = (
    transactions.groupby('hour')['is_fraud']
    .mean()
    .reset_index()
)
plt.figure(figsize=(8, 4))
plt.plot(hourly['hour'], hourly['is_fraud'], marker='o')
plt.xticks(range(0, 24))
plt.title('Доля мошенничества по часам дня')
plt.xlabel('Час дня')
plt.ylabel('Доля фрод-транзакций')
plt.tight_layout()
plt.show()

# График активности мошенников по дням недели
dow = (
    transactions.groupby('day_of_week')['is_fraud']
    .mean()
    .reset_index()
)
plt.figure(figsize=(6, 4))
plt.plot(dow['day_of_week'], dow['is_fraud'], marker='o')
plt.xticks(ticks=range(7), labels=['Пн','Вт','Ср','Чт','Пт','Сб','Вс'])
plt.title('Доля мошенничества по дням недели')
plt.xlabel('День недели')
plt.ylabel('Доля фрод-транзакций')
plt.tight_layout()
plt.show()
