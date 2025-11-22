import numpy as np
import pandas as pd

np.random.seed(42)

n_tx = 10_000
steps = np.random.randint(1, 31, n_tx)
types = np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT'], n_tx)
amounts = np.round(np.random.uniform(10, 5000, n_tx), 2)
name_origs = [f'C{id:08d}' for id in np.random.randint(1000, 999999, n_tx)]
name_dests = [f'C{id:08d}' for id in np.random.randint(1000, 999999, n_tx)]

# Метка фрода - всего 1%
is_fraud = np.zeros(n_tx)
fraud_idxs = np.random.choice(n_tx, n_tx // 100, replace=False)

# Логика: большие суммы/переводы часто фрод
for idx in fraud_idxs:
    is_fraud[idx] = 1
    amounts[idx] = np.round(np.random.uniform(3500, 5000), 2)
    types[idx] = np.random.choice(['TRANSFER', 'CASH_OUT'])

df = pd.DataFrame({
    'step': steps,
    'type': types,
    'amount': amounts,
    'nameOrig': name_origs,
    'nameDest': name_dests,
    'isFraud': is_fraud.astype(int),
})

df.to_csv('transactions.csv', index=False)
print("Сгенерировано и сохранено 10 000 транзакций в transactions.csv")
