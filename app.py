import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
from catboost import CatBoostClassifier
import time
import os

st.set_page_config(page_title="ForteBank Guard", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    /* Бордовый фон в стиле ForteBank */
    body {background-color: #a01f48 !important;}
    .stApp {background-color: #a01f48 !important;}
    /* Сайдбар: белый, тёмный шрифт */
    [data-testid="stSidebar"] {
        background: #fff !important;
        color: #222 !important;
    }
    /* Главная область: белые карточки, чёрный текст */
    .stMetric, .metric-title, .metric-value, .stDataFrame, .css-1v0mbdj, .css-12w0qpk, .css-1d391kg {
        background: #fff !important;
        color: #111 !important;
        border-radius: 14px !important;
        font-weight: bold !important;
    }
    /* Текст */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown, .stButton>button, label, .stTextInput>div>div>input, .stTextInput label, .stSliderLabel, .stSlider .st-cq, p, span, .st-c3, .st-ck, .stCaption {
        color: #111 !important;
    }
    /* Кнопки */
    .stButton>button {
        background: #fff !important;
        color: #1d1a22 !important;
        border: 1.5px solid #d3b1ac !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
    }
    /* Таблица */
    .stDataFrame table, .stDataFrame tr, .stDataFrame td, .stDataFrame th {
        color: #1a1a1a !important;
        background: #fff !important;
    }
    /* Бордовые мелкие акценты */
    hr, .css-105b1ln, .st-ac {
      border-color: #a01f48 !important;
    }
    </style>
    """, unsafe_allow_html=True
)
st.sidebar.title("ForteBank Guard")
risk_threshold = st.sidebar.slider("Порог риска", 0.0, 1.0, 0.5, 0.01)
retrain = st.sidebar.button("Переобучить модель")

CSV_FILE = 'transactions.csv'
MODEL_FILE = 'fraud_model.cbm'

# Переобучение по требованию
if retrain:
    with st.spinner("Обучаем модель..."):
        os.system('python train_model.py')
    st.sidebar.success("Модель переобучена!")

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_FILE)
    return model

@st.cache_data
def load_transactions():
    return pl.read_csv(CSV_FILE, separator=';')

def get_metrics(df, pred_probs):
    total = df.height
    frauds = np.sum(pred_probs > risk_threshold)
    mask = pl.Series(pred_probs > risk_threshold)
    filtered = df.filter(mask)
    saved = int(filtered['amount'].sum())
    return total, frauds, saved

# Загрузка
if not os.path.exists(CSV_FILE):
    st.error("Данных нет! Сначала запустите data_gen.py.")
    st.stop()
if not os.path.exists(MODEL_FILE):
    st.warning("Модели нет – запустите обучение! train_model.py")

    if st.button("Обучить сейчас"):
        with st.spinner("Обучаем модель..."):
            os.system('python train_model.py')
    st.stop()

model = load_model()
df = load_transactions()
# Для предсказания
X_app = df.drop(['isFraud', 'nameOrig', 'nameDest'])
X_app = X_app.with_columns([
    pl.col('type').cast(pl.Categorical)
])
proba = model.predict_proba(X_app.to_pandas())[:,1]
preds = (proba > risk_threshold).astype(int)

# Метрики (широкий верхний блок)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<span style="color:#111;font-size:20px;">Всего транзакций</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="color:#111;font-size:42px;font-weight:bold;">{df.height}</span>', unsafe_allow_html=True)
with col2:
    st.markdown('<span style="color:#111;font-size:20px;">Заблокировано атак</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="color:#111;font-size:42px;font-weight:bold;">{np.sum(preds)}</span>', unsafe_allow_html=True)
with col3:
    fraud_sum = int(df.filter(pl.Series(preds == 1))["amount"].sum())
    st.markdown(f'<span style="color:#111;font-size:42px;font-weight:bold;">{fraud_sum}</span>', unsafe_allow_html=True)

st.markdown('<h3 style="color:#111;">Последние транзакции</h3>', unsafe_allow_html=True)

def highlight_fraud(s):
    return ['background-color: #7f231a; color: #fff;' if v==1 else '' for v in s]

simulate = st.button("Симуляция потока")
if simulate:
    for i in range(20):
        subset = df.reverse().head(i+1).to_pandas()
        subset['FRAUD?'] = preds[::-1][:i+1]
        st.dataframe(subset.style.apply(highlight_fraud, subset=['FRAUD?']), height=420)
        time.sleep(0.2)
        st.rerun()
else:
    tbl = df.reverse().head(20).to_pandas()
    tbl['FRAUD?'] = preds[::-1][:20]
    st.dataframe(tbl.style.apply(highlight_fraud, subset=['FRAUD?']), height=420)

# Низ: график слева, explain справа
c1, c2 = st.columns(2)
with c1:
    fraud_time = df.with_columns([
        pl.Series('preds', preds)
    ]).filter(pl.col('preds') == 1)
    if fraud_time.height > 0:
        chart = fraud_time.group_by('step').agg(pl.sum('amount')).sort('step')
        fig = px.line(chart.to_pandas(), x='step', y='amount', title="Фрод по времени (сумма $)", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Фрода не обнаружено!")

with c2:
    st.markdown('<span style="color:#111;font-size:22px;font-weight:bold;">AI Объяснение</span>', unsafe_allow_html=True)
    if np.any(preds):
        st.markdown('<span style="color:#a01f48;font-size:20px;font-weight:bold;">⚠️ Блокировка: Аномально высокая сумма перевода для данного клиента</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#198c4c;font-size:18px;">Система работает в штатном режиме. Аномалий не выявлено.</span>', unsafe_allow_html=True)

st.caption("© ForteBank Guard Hackathon MVP")
