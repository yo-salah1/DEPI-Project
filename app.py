from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)

# 1. تحميل الملفات المحفوظة
model = joblib.load('sales_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('model_columns.json', 'r') as f:
    model_columns = json.load(f)

def preprocess_input(data_df):
    """
    دالة لتجهيز الداتا (سواء صف واحد أو ملف كامل)
    بنفس الخطوات اللي عملناها في الـ Notebook
    """
    # تحويل التاريخ واستخراج المعلومات
    if 'Date' in data_df.columns:
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        data_df['Year'] = data_df['Date'].dt.year
        data_df['Month'] = data_df['Date'].dt.month
        data_df['Day'] = data_df['Date'].dt.day
        data_df['WeekOfYear'] = data_df['Date'].dt.isocalendar().week
        data_df['DayOfWeek'] = data_df['Date'].dt.dayofweek + 1
        
        # حساب المدد الزمنية (مع التأكد من وجود الأعمدة)
        if 'CompetitionOpenSinceYear' in data_df.columns:
            data_df['CompetitionAge'] = data_df['Year'] - data_df['CompetitionOpenSinceYear']
            data_df['CompetitionAge'] = data_df['CompetitionAge'].apply(lambda x: x if x > 0 else 0)
            
        if 'Promo2SinceYear' in data_df.columns:
            data_df['Promo2Duration'] = data_df['Year'] - data_df['Promo2SinceYear']
            data_df['Promo2Duration'] = data_df['Promo2Duration'].apply(lambda x: x if x > 0 else 0)
            
        # حذف الأعمدة غير الضرورية (نفس اللي في Notebook)
        cols_to_drop = ['Store', 'Date', 'Customers', 'Open', 'PromoInterval', 
                        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                        'Promo2SinceWeek', 'Promo2SinceYear']
        data_df = data_df.drop(columns=cols_to_drop, errors='ignore')

    # One-Hot Encoding
    data_df = pd.get_dummies(data_df)

    # *** الخطوة السحرية ***
    # التأكد إن الأعمدة متطابقة مع أعمدة الموديل
    # لو في عمود ناقص (مثلا StoreType_b مش موجود في الانبوت) بنضيفه ونحط صفر
    data_df = data_df.reindex(columns=model_columns, fill_value=0)

    # Scaling
    data_scaled = scaler.transform(data_df)
    
    return data_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام البيانات من الفورم
        features = {
            'Store': 1, # قيمة افتراضية لأنه بيتحذف
            'DayOfWeek': int(request.form['DayOfWeek']),
            'Date': request.form['Date'],
            'Open': 1, # نفترض المحل مفتوح
            'Promo': int(request.form['Promo']),
            'StateHoliday': request.form['StateHoliday'],
            'SchoolHoliday': int(request.form['SchoolHoliday']),
            'StoreType': request.form['StoreType'],
            'Assortment': request.form['Assortment'],
            'CompetitionDistance': float(request.form['CompetitionDistance']),
            'CompetitionOpenSinceMonth': float(request.form.get('CompetitionOpenSinceMonth', 0)),
            'CompetitionOpenSinceYear': float(request.form.get('CompetitionOpenSinceYear', 0)),
            'Promo2': int(request.form['Promo2']),
            'Promo2SinceWeek': float(request.form.get('Promo2SinceWeek', 0)),
            'Promo2SinceYear': float(request.form.get('Promo2SinceYear', 0)),
            'PromoInterval': 'None'
        }
        
        # تحويل لـ DataFrame
        df = pd.DataFrame([features])
        
        # المعالجة والتوقع
        processed_data = preprocess_input(df)
        prediction = model.predict(processed_data)
        
        # إذا كنت مستخدم Log Transformation في التدريب، رجعها لأصلها
        # output = np.expm1(prediction[0]) 
        output = prediction[0] # لو مكنتش مستخدم Log
        return render_template('index.html', prediction_text=f'Sales Prediction: ${output:,.2f}')


    except Exception as e:
        return render_template('index.html', prediction_text=f'حدث خطأ: {str(e)}')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # قراءة الملف
        df = pd.read_csv(file)
        
        # حفظ نسخة من البيانات الأصلية للنتيجة
        original_df = df.copy()
        
        # المعالجة
        processed_data = preprocess_input(df)
        
        # التوقع
        predictions = model.predict(processed_data)
        
        # إضافة التوقعات للملف
        original_df['Predicted_Sales'] = predictions
        
        # تحويل لـ HTML للعرض (أو ممكن تحفظه كـ CSV وتخليه يحمله)
        return original_df.head(20).to_html(classes='table table-striped')

    except Exception as e:
        return f"Error processing file: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)