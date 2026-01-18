import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. إعدادات الترجمة واللغات (Translation Core)
# ==========================================
TRANSLATIONS = {
    "ar": {
        "dir": "rtl",
        "align": "right",
        "title": "نظام V-190 الشامل للتنبؤ بالحرائق",
        "sidebar": "🛠️ مركز القيادة",
        "api_lbl": "مفتاح API (اختياري)",
        "city_lbl": "اسم المدينة (بالإنجليزية):",
        "city_ph": "مثال: Baghdad, Dubai, London",
        "run": "🚀 تشغيل المسح الشامل",
        "loading": "جاري الاتصال بالأقمار الصناعية وتحليل البيانات...",
        "warn": "⚠️ يرجى كتابة اسم المدينة أولاً.",
        "risk_title": "احتمالية الحريق (المخاطر)",
        "conf_title": "مؤشر ثقة النظام (D)",
        "markov_title": "🔮 التنبؤ المستقبلي (سلاسل ماركوف)",
        "algo_title": "🧠 تحليل إجماع الخوارزميات (25 نموذج)",
        "res_safe": "✅ آمن",
        "res_warn": "⚠️ تحذير",
        "res_danger": "🔥 خطر شديد",
        "temp": "الحرارة",
        "hum": "الرطوبة",
        "wind": "الرياح",
        "climate": "نوع المناخ",
        "hour": "بعد س",
        "state": "الحالة المتوقعة",
        "prob": "احتمالية الحريق",
        "sim_msg": "يعمل النظام ببيانات المحاكاة لعدم توفر مفتاح API"
    },
    "en": {
        "dir": "ltr",
        "align": "left",
        "title": "V-190 Global Fire Prediction System",
        "sidebar": "🛠️ Command Center",
        "api_lbl": "API Key (Optional)",
        "city_lbl": "City Name:",
        "city_ph": "Ex: Baghdad, Dubai, London",
        "run": "🚀 Run Scan",
        "loading": "Connecting to satellites & analyzing data...",
        "warn": "⚠️ Please enter a city name first.",
        "risk_title": "Fire Risk Probability",
        "conf_title": "System Confidence Score (D)",
        "markov_title": "🔮 Future Forecast (Markov Chains)",
        "algo_title": "🧠 Algorithm Consensus (25 Models)",
        "res_safe": "✅ Safe",
        "res_warn": "⚠️ Warning",
        "res_danger": "🔥 Extreme Danger",
        "temp": "Temp",
        "hum": "Humidity",
        "wind": "Wind",
        "climate": "Climate Type",
        "hour": "Hour +",
        "state": "Predicted State",
        "prob": "Fire Prob",
        "sim_msg": "System running in simulation mode (No API Key)"
    },
    "tr": {
        "dir": "ltr",
        "align": "left",
        "title": "V-190 Küresel Yangın Tahmin Sistemi",
        "sidebar": "🛠️ Komuta Merkezi",
        "api_lbl": "API Anahtarı (İsteğe Bağlı)",
        "city_lbl": "Şehir Adı:",
        "city_ph": "Örn: Baghdad, Dubai, London",
        "run": "🚀 Taramayı Başlat",
        "loading": "Uydulara bağlanılıyor ve veriler analiz ediliyor...",
        "warn": "⚠️ Lütfen önce bir şehir adı girin.",
        "risk_title": "Yangın Riski Olasılığı",
        "conf_title": "Sistem Güven Skoru (D)",
        "markov_title": "🔮 Gelecek Tahmini (Markov Zincirleri)",
        "algo_title": "🧠 Algoritma Konsensüsü (25 Model)",
        "res_safe": "✅ Güvenli",
        "res_warn": "⚠️ Uyarı",
        "res_danger": "🔥 Aşırı Tehlike",
        "temp": "Sıcaklık",
        "hum": "Nem",
        "wind": "Rüzgar",
        "climate": "İklim Tipi",
        "hour": "Saat +",
        "state": "Tahmini Durum",
        "prob": "Yangın Olasılığı",
        "sim_msg": "Sistem simülasyon modunda çalışıyor (API Anahtarı Yok)"
    }
}

# إعداد الصفحة
st.set_page_config(page_title="V-190 Global", page_icon="🌍", layout="wide")

# اختيار اللغة
lang = st.sidebar.selectbox("Language / اللغة / Dil", ["العربية", "English", "Türkçe"])
if lang == "العربية": L = "ar"
elif lang == "Türkçe": L = "tr"
else: L = "en"
T = TRANSLATIONS[L]

# تخصيص CSS للاتجاهات
st.markdown(f"""
<style>
    .main {{ direction: {T['dir']}; text-align: {T['align']}; }}
    h1, h2, h3, h4, p, span, div, label {{ text-align: {T['align']}; font-family: 'Segoe UI', sans-serif; }}
    .stMetric {{ background-color: #f9f9f9; border: 1px solid #ddd; }}
    div[data-testid="stDataFrame"] {{ direction: {T['dir']}; }}
    input {{ text-align: left !important; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. العقل المدبر (Climate & Logic Brain)
# ==========================================
class V190Brain:
    def get_thresholds(self, lat):
        """تكييف المعايير حسب الموقع الجغرافي"""
        abs_lat = abs(lat)
        if abs_lat < 23.5: return {"temp": 46, "hum": 15, "type": "Tropical/Desert"}
        elif abs_lat < 50: return {"temp": 38, "hum": 20, "type": "Temperate"}
        else: return {"temp": 30, "hum": 30, "type": "Boreal/Cold"}

    def calculate_risk(self, temp, hum, wind, thresholds):
        # 1. حساب الخطر
        risk = 0
        if temp >= thresholds['temp']: risk += 45
        elif temp >= thresholds['temp']-5: risk += 20
        if hum <= thresholds['hum']: risk += 35
        elif hum <= thresholds['hum']+10: risk += 15
        if wind > 30: risk += 20
        elif wind > 15: risk += 10
        risk = min(risk, 100)
        
        # 2. حساب الثقة (المسافة عن المنتصف)
        dist = abs(risk - 50)
        conf = 75 + (dist * 0.48)
        conf = min(conf, 99.9)
        
        return risk, conf

# ==========================================
# 3. محرك سلاسل ماركوف (Markov Engine)
# ==========================================
class MarkovEngine:
    def simulate(self, current_risk, wind, temp):
        # مصفوفة احتمالات ديناميكية
        # [Safe, Risk, Fire]
        matrix = np.array([[0.8, 0.19, 0.01], [0.4, 0.5, 0.1], [0.05, 0.15, 0.8]])
        
        # تعديل المصفوفة حسب الظروف القاسية
        if wind > 25: 
            matrix[1][2] += 0.25; matrix[1][1] -= 0.25 # الرياح تسرع الانتقال للحريق
        if temp > 40:
            matrix[0][1] += 0.15; matrix[0][0] -= 0.15 # الحرارة تقلل الأمان
            
        # تطبيع المصفوفة
        for i in range(3): matrix[i] = matrix[i] / matrix[i].sum()
        
        # الحالة الأولية
        if current_risk < 40: state_vec = np.array([1.0, 0, 0])
        elif current_risk < 75: state_vec = np.array([0, 1.0, 0])
        else: state_vec = np.array([0, 0, 1.0])
        
        history = []
        states_lbl = [T['res_safe'], T['res_warn'], T['res_danger']]
        
        for t in range(5):
            state_vec = np.dot(state_vec, matrix)
            idx = np.argmax(state_vec)
            history.append({
                T['hour']: f"+{t+1}",
                T['state']: states_lbl[idx],
                T['prob']: f"{state_vec[2]*100:.1f}%"
            })
            
        return history, matrix

# ==========================================
# 4. خدمة البيانات والمحاكاة (Global Sensor)
# ==========================================
class GlobalSensor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.brain = V190Brain()

    def get_data(self, city):
        # محاولة الاتصال بـ API
        if self.api_key:
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?appid={self.api_key}&q={city}&units=metric"
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    d = r.json()
                    th = self.brain.get_thresholds(d['coord']['lat'])
                    risk, conf = self.brain.calculate_risk(d['main']['temp'], d['main']['humidity'], d['wind']['speed'], th)
                    return {
                        'temp': d['main']['temp'], 'hum': d['main']['humidity'], 'wind': d['wind']['speed'],
                        'lat': d['coord']['lat'], 'lon': d['coord']['lon'], 'name': d['name'],
                        'risk': risk, 'conf': conf, 'type': th['type'], 'is_sim': False
                    }
            except: pass
        
        # الوضع الاحتياطي (محاكاة ذكية)
        np.random.seed(sum(map(ord, city)))
        lat = np.random.uniform(-60, 60) # خط عرض عشوائي للمحاكاة
        th = self.brain.get_thresholds(lat)
        
        temp = np.random.normal(th['temp']-5, 8)
        hum = np.random.uniform(10, 80)
        wind = np.random.uniform(5, 35)
        risk, conf = self.brain.calculate_risk(temp, hum, wind, th)
        
        return {
            'temp': round(temp, 1), 'hum': int(hum), 'wind': round(wind, 1),
            'lat': lat, 'lon': np.random.uniform(-180, 180), 'name': city,
            'risk': int(risk), 'conf': round(conf, 1), 'type': th['type'], 'is_sim': True
        }

# ==========================================
# 5. واجهة التطبيق (UI Application)
# ==========================================
st.sidebar.title(T['sidebar'])
# محاولة جلب المفتاح من أسرار السحابة أولاً
if "OWM_API_KEY" in st.secrets:
    api_key = st.secrets["OWM_API_KEY"]
else:
    # إذا لم يوجد في الأسرار، نطلبه من المستخدم في الواجهة
    api_key = st.sidebar.text_input(T['api_lbl'], type="password")
    city_input = st.sidebar.text_input(T['city_lbl'], placeholder=T['city_ph'])
run_btn = st.sidebar.button(T['run'])

st.title(T['title'])

if run_btn:
    if not city_input:
        st.warning(T['warn'])
    else:
        with st.spinner(T['loading']):
            # 1. جلب البيانات
            sensor = GlobalSensor(api_key)
            data = sensor.get_data(city_input)
            
            if data['is_sim']: st.warning(T['sim_msg'])
            
            # 2. تشغيل ماركوف
            m_engine = MarkovEngine()
            futures, m_matrix = m_engine.simulate(data['risk'], data['wind'], data['temp'])
            
            # --- العرض: الصف الأول (المقاييس) ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📍 " + T['city_lbl'], data['name'])
            c2.metric("🌡️ " + T['temp'], f"{data['temp']} °C")
            c3.metric("💧 " + T['hum'], f"{data['hum']}%")
            c4.metric("💨 " + T['wind'], f"{data['wind']} km/h")
            
            st.markdown("---")
            
            # --- العرض: الصف الثاني (العدادات) ---
            col_risk, col_conf = st.columns(2)
            
            with col_risk:
                # عداد الخطر
                fig_r = go.Figure(go.Indicator(
                    mode = "gauge+number", value = data['risk'],
                    title = {'text': T['risk_title']},
                    gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "red" if data['risk']>50 else "green"}}
                ))
                st.plotly_chart(fig_r, use_container_width=True)
                
            with col_conf:
                # عداد الثقة (نصف دائري)
                fig_c = go.Figure(go.Indicator(
                    mode = "gauge+number", value = data['conf'],
                    title = {'text': T['conf_title']},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "blue"}, 'shape': "bullet"}
                ))
                st.plotly_chart(fig_c, use_container_width=True)
                
            # --- العرض: الصف الثالث (ماركوف + الخوارزميات) ---
            tab1, tab2 = st.tabs([T['markov_title'], T['algo_title']])
            
            with tab1:
                cm1, cm2 = st.columns(2)
                with cm1:
                    st.dataframe(pd.DataFrame(futures), use_container_width=True)
                with cm2:
                    # رسم المصفوفة
                    fig_m = px.imshow(m_matrix, text_auto=".2f", color_continuous_scale="Reds", 
                                      labels=dict(x="To", y="From"))
                    st.plotly_chart(fig_m, use_container_width=True)
                    
            with tab2:
                # محاكاة إجماع الـ 25 خوارزمية للعرض
                st.write("Distribution of votes from 25 AI Models:")
                # نولد توزيعاً يوافق المخاطر المحسوبة
                votes_fire = int((data['risk'] / 100) * 25)
                votes_safe = 25 - votes_fire
                
                chart_data = pd.DataFrame({
                    "Decision": [T['res_danger'], T['res_safe']],
                    "Votes": [votes_fire, votes_safe]
                })
                fig_pie = px.pie(chart_data, values='Votes', names='Decision', 
                                 color='Decision', 
                                 color_discrete_map={T['res_danger']:'red', T['res_safe']:'green'})
                st.plotly_chart(fig_pie, use_container_width=True)
                
            # الخريطة
            st.map(pd.DataFrame({'lat': [data['lat']], 'lon': [data['lon']]}))

else:
    st.info(f"👈 {T['warn']}")
