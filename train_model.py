import polars as pl
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils import resample
import joblib

TX_FILE = 'transactions.csv'
BEHAVIOR_FILE = 'behavior_fixed.csv'
MERGED_FILE = 'transactions_with_behavior.csv'

# Загрузка транзакций
print('Загрузка транзакций...')
df_tx = pl.read_csv(TX_FILE, separator=';')
# Загрузка поведенческих паттернов с обработкой нечисловых значений как null
print('Загрузка паттернов клиентов...')
df_beh = pl.read_csv(
    BEHAVIOR_FILE,
    separator=';',
    null_values=['01.май', 'err', 'NONE', 'None', 'NaN', 'nan', ''],
    infer_schema_length=10000,
    ignore_errors=True
)

# Приводим к строке внутренние id для join
print('Преобразование id-клиентов...')
df_tx = df_tx.with_columns(pl.col('nameOrig').cast(pl.Utf8))
df_beh = df_beh.with_columns(pl.col('cst_dim_id').cast(pl.Utf8))

# Соединение по полю клиента
print('Объединение транзакций и паттернов...')
df_merged = df_tx.join(df_beh, left_on='nameOrig', right_on='cst_dim_id', how='left')
if 'cst_dim_id' in df_merged.columns:
    df_merged = df_merged.drop(['cst_dim_id'])

# Опционально: сохраняем объединённый датасет для контроля
df_merged.write_csv(MERGED_FILE, separator=';')
print('Объединённый датасет сохранён в', MERGED_FILE)

# Далее обучение CatBoost
# Категориальные переменные
cat_features = ['type']
# Отделим фичи и target. Уберём nameOrig и nameDest (они id, их не считаем полезными фичами)
exclude_cols = ['isFraud', 'nameOrig', 'nameDest']
X = df_merged.drop(exclude_cols).to_pandas()
X['type'] = X['type'].astype('category')
y = df_merged['isFraud'].to_numpy()

# Балансировка: undersample неподозрительных
X_major = X[y == 0]
X_minor = X[y == 1]
y_major = y[y == 0]
y_minor = y[y == 1]
X_major_down, y_major_down = resample(X_major, y_major,
                                      replace=False,
                                      n_samples=len(y_minor),
                                      random_state=42)
X_bal = pd.concat([X_major_down, X_minor])
y_bal = np.concatenate([y_major_down, y_minor])

print('Обучение CatBoost...')
model = CatBoostClassifier(iterations=150, random_seed=42, cat_features=cat_features, verbose=0)
model.fit(X_bal, y_bal)

model.save_model('fraud_model.cbm')
print('Модель обучена и сохранена в fraud_model.cbm')
