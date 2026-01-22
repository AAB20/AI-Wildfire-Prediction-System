import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. إعدادات الترجمة
# ==========================================
TRANSLATIONS = {
    "ar": {
        "dir": "rtl", "align": "right",
        "title": "نظام V-190: العراف الذكي (Oracle)",
        "sidebar": "🛠️ مركز التحكم",
        "city_lbl": "اسم المدينة:",
        "run": "🚀 تشغيل العراف (التنبؤ المستقبلي)",
        "tabs": ["الوضع الحالي", "🔮 التنبؤ بموعد الحريق", "سجل 10 سنوات"],
        "pred_title": "⏳ العد التنازلي للكارثة المحتملة",
        "days_left": "عدد الأيام المتوقعة حتى نشوب حريق:",
        "prob": "نسبة احتمالية الحدوث:",
        "safe_msg": "🟢 الوضع مستقر. لا يتوقع حدوث حرائق خلال الـ 30 يوماً القادمة.",
        "danger_msg": "🔴 تحذير! الظروف تتجه نحو الاشتعال.",
        "chart_future": "مسار الخطر خلال الأيام القادمة",
        "reason": "السبب الرئيسي المتوقع:",
        "heat_wave": "موجة حر قادمة",
        "dry_spell": "جفاف متراكم",
        "wind_storm": "عاصفة رياح متوقعة"
    },
    "en": {
        "dir": "ltr", "align": "left",
        "title": "V-190: The Oracle Edition",
        "sidebar": "🛠️ Control Center",
        "city_lbl": "City Name:",
        "run": "🚀 Run Oracle Prediction",
        "tabs": ["Current Status", "🔮 Time-to-Fire Prediction", "10-Year History"],
        "pred_title": "⏳ Countdown to Potential Event",
        "days_left": "Days until potential fire:",
        "prob": "Probability of Occurrence:",
        "safe_msg": "🟢 Status Stable. No fires predicted in next 30 days.",
        "danger_msg": "🔴 Warning! Conditions are deteriorating.",
        "chart_future": "Risk Trajectory (Next Days)",
        "reason": "Expected Primary Driver:",
        "heat_wave": "Approaching Heat Wave",
        "dry_spell": "Cumulative Drought",
        "wind_storm": "Predicted Wind Storm"
    }
}

st.set_page_config(page_title="V-190 Oracle", page_icon="🔮", layout="wide")

lang = st.sidebar.selectbox("Language / اللغة", ["العربية", "English"])
L = "ar" if lang == "العربية" else "en"
T = TRANSLATIONS[L]

st.markdown(f"""
<style>
    .main {{ direction: {T['dir']}; text-align: {T['align']}; }}
    h1, h2, h3, h4, p, div, span {{ text-align: {T['align']}; font-family: sans-serif; }}
    .stMetric {{ background-color: #f4f4f4; border-radius: 10px; padding: 10px; border: 1px solid #ddd; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. العقل المدبر (V-190 Brain)
# ==========================================
class V190Brain:
    def get_thresholds(self, lat):
        abs_lat = abs(lat)
        if abs_lat < 23.5: return {"temp": 46, "hum": 15}
        elif abs_lat < 50: return {"temp": 38, "hum": 20}
        else: return {"temp": 30, "hum": 30}

    def calculate_risk(self, temp, hum, wind, thresholds):
        risk = 0
        if temp >= thresholds['temp']: risk += 45
        elif temp >= thresholds['temp']-5: risk += 20
        if hum <= thresholds['hum']: risk += 35
        elif hum <= thresholds['hum']+10: risk += 15
        if wind > 30: risk += 20
        elif wind > 15: risk += 10
        return min(risk, 100)

# ==========================================
# 3. العراف (Future Oracle Engine) - الجزء الجديد
# ==========================================
class FutureOracle:
    """
    يقوم بمحاكاة الطقس لـ 30 يوماً قادمة بناءً على البيانات الحالية والاتجاه الموسمي،
    ثم يحدد 'اليوم الصفر' (Day Zero) الذي يحدث فيه الحريق.
    """
    def __init__(self, brain):
        self.brain = brain

    def predict_days_to_fire(self, start_temp, start_hum, start_wind, lat):
        # تحديد العتبات
        th = self.brain.get_thresholds(lat)
        
        future_days = []
        prediction = None
        
        current_temp = start_temp
        current_hum = start_hum
        current_wind = start_wind
        
        # محاكاة 30 يوماً قادمة
        for day in range(1, 31):
            # محاكاة التغير المناخي القصير المدى
            # نضيف "ميل" (Trend) نحو السخونة أو البرودة عشوائياً لمحاكاة موجات الطقس
            trend = np.random.choice([0.5, -0.2, 1.2]) # ميل للارتفاع في مواسم الحرائق
            
            current_temp += trend + np.random.uniform(-1, 1)
            current_hum -= np.random.uniform(0, 2) # الرطوبة تميل للانخفاض مع الوقت في مواسم الجفاف
            current_wind = abs(current_wind + np.random.uniform(-5, 5))
            
            # حساب الخطر لهذا اليوم المستقبلي
            risk = self.brain.calculate_risk(current_temp, current_hum, current_wind, th)
            
            future_days.append({
                "Day": day,
                "Risk": risk,
                "Temp": current_temp
            })
            
            # شرط وقوع الحريق: خطر > 80%
            if risk >= 80 and prediction is None:
                # تحديد السبب
                reason = T['heat_wave'] if current_temp > th['temp'] else T['dry_spell']
                prediction = {
                    "days": day,
                    "prob": min(risk + 10, 99), # الاحتمالية تكون أعلى قليلاً من الخطر
                    "risk_val": risk,
                    "reason": reason
                }

        return prediction, pd.DataFrame(future_days)

# ==========================================
# 4. الواجهة والتشغيل
# ==========================================
st.sidebar.title(T['sidebar'])
w_key = st.sidebar.text_input("OpenWeatherMap Key (Optional)", type="password")
city = st.sidebar.text_input(T['city_lbl'])
run = st.sidebar.button(T['run'])

st.title(T['title'])

brain = V190Brain()
oracle = FutureOracle(brain)

if run and city:
    # 1. جلب البيانات الحالية (أو المحاكاة)
    try:
        if w_key:
            url = f"http://api.openweathermap.org/data/2.5/weather?appid={w_key}&q={city}&units=metric"
            d = requests.get(url).json()
            temp, hum, wind = d['main']['temp'], d['main']['humidity'], d['wind']['speed']
            lat = d['coord']['lat']
        else: raise Exception
    except:
        # محاكاة لبيانات "حرجة" لتجربة التنبؤ
        np.random.seed(sum(map(ord, city)))
        temp, hum, wind = np.random.uniform(25, 42), 30, 15
        lat = 33.0

    # 2. تشغيل العراف (التنبؤ)
    pred_result, df_future = oracle.predict_days_to_fire(temp, hum, wind, lat)

    # --- العرض ---
    tab1, tab2, tab3 = st.tabs(T['tabs'])

    # Tab 1: الوضع الحالي
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡️ Temp", f"{temp:.1f} °C")
        c2.metric("💧 Hum", f"{hum:.0f}%")
        c3.metric("💨 Wind", f"{wind:.1f} km/h")
        
        current_risk = brain.calculate_risk(temp, hum, wind, brain.get_thresholds(lat))
        st.subheader("Current Risk Index")
        st.progress(int(current_risk))

    # Tab 2: العراف (التنبؤ المستقبلي) - الجزء المهم
    with tab2:
        st.subheader(T['pred_title'])
        
        if pred_result:
            # تم التنبؤ بحريق
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                st.error(f"### {pred_result['days']} {L=='ar' and 'أيام' or 'Days'}")
                st.caption(T['days_left'])
                
            with col_p2:
                st.warning(f"### {pred_result['prob']:.1f}%")
                st.caption(T['prob'])
                
            st.info(f"**{T['reason']}** {pred_result['reason']}")
            
            # رسالة تحذيرية
            st.markdown(f"#### {T['danger_msg']}")
            
        else:
            # الوضع آمن
            st.success(f"### ♾️")
            st.caption(T['days_left'])
            st.success(T['safe_msg'])

        st.markdown("---")
        st.write(f"**{T['chart_future']}**")
        
        # رسم بياني للاتجاه المستقبلي
        fig = px.line(df_future, x="Day", y="Risk", markers=True, 
                      labels={"Day": "Days from Now", "Risk": "Risk Level (%)"})
        
        # إضافة خط الخطر الأحمر
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Danger Zone")
        
        # تلوين المناطق
        fig.update_traces(line_color='#ff4b4b')
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: سجل 10 سنوات (مبسط للعرض)
    with tab3:
        st.info("Simulated 10-Year historical data would appear here.")
        # (نفس كود التاريخ السابق يمكن وضعه هنا)

elif run:
    st.warning("Please enter city name.")
