import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.express as px
import plotly.graph_objects as go

# --- ML & DL Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

# ==========================================
# 1. قاموس الترجمة (Translation Center)
# ==========================================
TRANSLATIONS = {
    "ar": {
        "dir": "rtl",
        "title": "نظام V-190 الذكي للتنبؤ بالحرائق",
        "sidebar_title": "🛠️ مركز التحكم",
        "api_lbl": "مفتاح API (اختياري)",
        "city_lbl": "اسم المدينة:",
        "city_ph": "مثال: Baghdad, Istanbul, London",
        "run_btn": "🚀 تشغيل التحليل",
        "loading": "جاري تحليل البيانات عبر الأقمار الصناعية...",
        "warn_city": "يرجى إدخال اسم المدينة أولاً.",
        "sim_mode": "⚠️ وضع المحاكاة (بدون API)",
        "metric_city": "المدينة",
        "metric_temp": "الحرارة",
        "metric_hum": "الرطوبة",
        "metric_wind": "الرياح",
        "risk_lbl": "احتمالية الحريق (AI)",
        "status_safe": "✅ آمن",
        "status_risk": "⚠️ خطر",
        "status_fire": "🔥 حريق محتمل",
        "markov_title": "🔮 محاكاة المستقبل (سلاسل ماركوف)",
        "markov_desc": "توقعات تطور الحالة خلال الـ 5 ساعات القادمة.",
        "matrix_title": "📊 مصفوفة الاحتمالات الانتقالية",
        "tab_res": "النتائج الحالية",
        "tab_sim": "المحاكاة المستقبلية",
        "col_hour": "الساعة",
        "col_state": "الحالة المتوقعة",
        "col_prob_fire": "احتمالية الحريق",
        "states": ["آمن", "خطر", "حريق"]
    },
    "en": {
        "dir": "ltr",
        "title": "V-190 Hyper-Intelligence Fire System",
        "sidebar_title": "🛠️ Control Center",
        "api_lbl": "API Key (Optional)",
        "city_lbl": "City Name:",
        "city_ph": "Ex: Baghdad, Istanbul, London",
        "run_btn": "🚀 Run Analysis",
        "loading": "Analyzing satellite data...",
        "warn_city": "Please enter a city name first.",
        "sim_mode": "⚠️ Simulation Mode (No API)",
        "metric_city": "City",
        "metric_temp": "Temperature",
        "metric_hum": "Humidity",
        "metric_wind": "Wind Speed",
        "risk_lbl": "Fire Risk (AI)",
        "status_safe": "✅ Safe",
        "status_risk": "⚠️ Risk",
        "status_fire": "🔥 Potential Fire",
        "markov_title": "🔮 Future Simulation (Markov Chains)",
        "markov_desc": "State evolution forecast for the next 5 hours.",
        "matrix_title": "📊 Transition Probability Matrix",
        "tab_res": "Current Results",
        "tab_sim": "Future Simulation",
        "col_hour": "Hour",
        "col_state": "Predicted State",
        "col_prob_fire": "Fire Probability",
        "states": ["Safe", "Risk", "Fire"]
    },
    "tr": {
        "dir": "ltr",
        "title": "V-190 Hiper-Zeka Yangın Sistemi",
        "sidebar_title": "🛠️ Kontrol Merkezi",
        "api_lbl": "API Anahtarı (İsteğe bağlı)",
        "city_lbl": "Şehir Adı:",
        "city_ph": "Örn: Baghdad, Istanbul, London",
        "run_btn": "🚀 Analizi Başlat",
        "loading": "Uydu verileri analiz ediliyor...",
        "warn_city": "Lütfen önce bir şehir adı girin.",
        "sim_mode": "⚠️ Simülasyon Modu (API Yok)",
        "metric_city": "Şehir",
        "metric_temp": "Sıcaklık",
        "metric_hum": "Nem",
        "metric_wind": "Rüzgar",
        "risk_lbl": "Yangın Riski (YZ)",
        "status_safe": "✅ Güvenli",
        "status_risk": "⚠️ Riskli",
        "status_fire": "🔥 Olası Yangın",
        "markov_title": "🔮 Gelecek Simülasyonu (Markov Zincirleri)",
        "markov_desc": "Önümüzdeki 5 saat için durum tahmini.",
        "matrix_title": "📊 Geçiş Olasılık Matrisi",
        "tab_res": "Mevcut Sonuçlar",
        "tab_sim": "Gelecek Simülasyonu",
        "col_hour": "Saat",
        "col_state": "Tahmini Durum",
        "col_prob_fire": "Yangın Olasılığı",
        "states": ["Güvenli", "Riskli", "Yangın"]
    }
}

# ==========================================
# 2. إعداد الصفحة واللغة
# ==========================================
st.set_page_config(page_title="V-190 Global", page_icon="🌍", layout="wide")

# اختيار اللغة في الشريط الجانبي (أول شيء)
lang_choice = st.sidebar.selectbox("Language / اللغة / Dil", ["العربية", "English", "Türkçe"])

# تعيين كود اللغة
if lang_choice == "العربية": lang_code = "ar"
elif lang_choice == "Türkçe": lang_code = "tr"
else: lang_code = "en"

# تحميل النصوص بناءً على الاختيار
T = TRANSLATIONS[lang_code]

# حقن CSS لتغيير الاتجاه (RTL/LTR)
st.markdown(f"""
<style>
    .main {{ direction: {T['dir']}; text-align: {'right' if T['dir'] == 'rtl' else 'left'}; }}
    h1, h2, h3, h4, p, span, div, label {{ 
        text-align: {'right' if T['dir'] == 'rtl' else 'left'}; 
        font-family: 'Segoe UI', Tahoma, sans-serif; 
    }}
    .stMetric {{ background-color: #f8f9fa; border: 1px solid #ddd; }}
    div[data-testid="stDataFrame"] {{ direction: {T['dir']}; }}
    /* محاذاة المدخلات دائماً لليسار لأنها إنجليزية غالباً */
    input {{ text-align: left !important; }} 
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. الفئات الأساسية (AI & Markov Engines)
# ==========================================

class MarkovSimulator:
    def __init__(self, lang_states):
        self.states = lang_states # يأخذ الأسماء من اللغة المختارة
        
    def get_matrix(self, wind, temp):
        # مصفوفة الاحتمالات (فيزياء بحتة - لا تتغير بتغير اللغة)
        matrix = np.array([
            [0.80, 0.19, 0.01],
            [0.40, 0.50, 0.10],
            [0.05, 0.15, 0.80]
        ])
        # تأثير الرياح والحرارة
        if wind > 20:
            matrix[1][2] += 0.30; matrix[1][1] -= 0.20; matrix[1][0] -= 0.10
        if temp > 40:
            matrix[0][1] += 0.10; matrix[0][0] -= 0.10
        
        # التطبيع
        for i in range(3): matrix[i] = matrix[i] / np.sum(matrix[i])
        return matrix

    def simulate(self, current_risk, wind, temp, hours=5):
        if current_risk < 40: curr = 0
        elif current_risk < 75: curr = 1
        else: curr = 2
        
        matrix = self.get_matrix(wind, temp)
        current_vec = np.zeros(3); current_vec[curr] = 1.0
        results = []
        
        for t in range(hours):
            next_vec = np.dot(current_vec, matrix)
            idx = np.argmax(next_vec)
            results.append({
                T["col_hour"]: f"+{t+1}",
                T["col_state"]: self.states[idx],
                T["col_prob_fire"]: f"{next_vec[2]*100:.1f}%"
            })
            current_vec = next_vec
        return results, matrix

class AI_Engine:
    @st.cache_resource
    def train_models(_self):
        # توليد بيانات وتدريب سريع (كما في الكود السابق)
        np.random.seed(42)
        X = np.random.rand(1000, 3) # Temp, Hum, Wind
        y = (X[:, 0]*0.5 + (1-X[:, 1])*0.4 + X[:, 2]*0.3 > 0.6).astype(int)
        
        # نموذج RF
        rf = RandomForestClassifier(n_estimators=10)
        rf.fit(X, y)
        
        # نموذج LSTM بسيط (محاكاة)
        # في التطبيق الحقيقي نستخدم Tensorflow هنا
        return rf

class WeatherService:
    def get_weather(self, api_key, city):
        if not api_key: return None, "NoKey"
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?appid={api_key}&q={city}&units=metric"
            r = requests.get(url, timeout=3)
            if r.status_code == 200: return r.json(), None
            return None, "Error"
        except: return None, "ConnError"

# ==========================================
# 4. واجهة التطبيق (UI Logic)
# ==========================================

st.sidebar.title(T["sidebar_title"])
api_key = st.sidebar.text_input(T["api_lbl"], type="password")
city_input = st.sidebar.text_input(T["city_lbl"], placeholder=T["city_ph"])
run_btn = st.sidebar.button(T["run_btn"])

st.title(T["title"])

# تهيئة المحركات
ai_engine = AI_Engine()
model = ai_engine.train_models()
markov = MarkovSimulator(T["states"]) # نمرر أسماء الحالات باللغة المختارة
weather_srv = WeatherService()

if run_btn:
    if not city_input:
        st.warning(T["warn_city"])
    else:
        with st.spinner(T["loading"]):
            # 1. Weather Data
            raw_data, err = weather_srv.get_weather(api_key, city_input)
            
            if err:
                st.warning(T["sim_mode"])
                np.random.seed(len(city_input))
                w = {
                    'temp': np.random.uniform(20, 45),
                    'hum': np.random.uniform(10, 60),
                    'wind': np.random.uniform(5, 30),
                    'name': city_input
                }
            else:
                w = {
                    'temp': raw_data["main"]["temp"],
                    'hum': raw_data["main"]["humidity"],
                    'wind': raw_data["wind"]["speed"],
                    'name': raw_data["name"]
                }

            # 2. AI Prediction
            # تطبيع بسيط للمحاكاة
            inputs = np.array([[w['temp']/50, w['hum']/100, w['wind']/40]])
            risk_prob = model.predict_proba(inputs)[0][1] * 100
            
            # 3. Markov Simulation
            future_data, trans_matrix = markov.simulate(risk_prob, w['wind'], w['temp'])
            
            # --- Display Tabs ---
            tab1, tab2 = st.tabs([T["tab_res"], T["tab_sim"]])
            
            with tab1:
                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(T["metric_city"], w['name'])
                c2.metric(T["metric_temp"], f"{w['temp']:.1f} °C")
                c3.metric(T["metric_hum"], f"{int(w['hum'])}%")
                c4.metric(T["metric_wind"], f"{w['wind']} km/h")
                
                st.markdown("---")
                
                # Gauge Chart
                g_col, txt_col = st.columns([1, 2])
                with g_col:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_prob,
                        title = {'text': T["risk_lbl"]},
                        gauge = {'axis': {'range': [None, 100]},
                                 'bar': {'color': "red" if risk_prob > 50 else "green"}}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                with txt_col:
                    st.subheader(T["risk_lbl"])
                    if risk_prob > 75:
                        st.error(f"### {T['status_fire']}")
                    elif risk_prob > 40:
                        st.warning(f"### {T['status_risk']}")
                    else:
                        st.success(f"### {T['status_safe']}")

            with tab2:
                st.subheader(T["markov_title"])
                st.write(T["markov_desc"])
                
                m_col1, m_col2 = st.columns(2)
                
                with m_col1:
                    st.table(pd.DataFrame(future_data))
                    
                with m_col2:
                    st.write(f"**{T['matrix_title']}**")
                    # رسم المصفوفة باستخدام أسماء الحالات المترجمة
                    fig_hm = px.imshow(trans_matrix,
                                       x=T["states"],
                                       y=T["states"],
                                       text_auto=".2f",
                                       color_continuous_scale="Blues",
                                       labels=dict(color="Prob"))
                    st.plotly_chart(fig_hm, use_container_width=True)

else:
    # شاشة البداية
    st.info("👈 " + ("الرجاء اختيار المدينة والبدء" if lang_code == 'ar' else "Please select a city to start" if lang_code == 'en' else "Lütfen başlamak için bir şehir seçin"))
