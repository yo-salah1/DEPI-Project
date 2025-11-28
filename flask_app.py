from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)

# Load artifacts once at startup
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    
    if request.method == 'POST':
        try:
            # Receive form data
            data = {
                'Date': pd.to_datetime(request.form['Date']),
                'StoreType': request.form['StoreType'],
                'Assortment': request.form['Assortment'],
                'Promo': int(request.form['Promo']),
                'StateHoliday': request.form['StateHoliday'],
                'SchoolHoliday': int(request.form['SchoolHoliday']),
                'CompetitionDistance': float(request.form['CompetitionDistance']),
                'CompetitionOpenSinceYear': float(request.form.get('CompetitionOpenSinceYear', 0)),
                'Promo2': int(request.form['Promo2']),
                'Promo2SinceYear': float(request.form.get('Promo2SinceYear', 0))
            }
            
            df = pd.DataFrame([data])

            # Feature Engineering
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['DayOfMonth'] = df['Date'].dt.day
            df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
            df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
            
            df['CompetitionAge'] = df['Year'] - df['CompetitionOpenSinceYear']
            df['CompetitionAge'] = df['CompetitionAge'].apply(lambda x: x if x > 0 else 0)
            
            df['Promo2Duration'] = df['Year'] - df['Promo2SinceYear']
            df['Promo2Duration'] = df['Promo2Duration'].apply(lambda x: x if x > 0 else 0)
            
            df['CompetitionDistance'] = np.log1p(df['CompetitionDistance'])

            # Drop unnecessary columns
            cols_to_drop = ['Date', 'CompetitionOpenSinceYear', 'Promo2SinceYear']
            df = df.drop(columns=cols_to_drop, errors='ignore')

            # Encoding & Alignment
            df_encoded = pd.get_dummies(df)
            
            final_df = pd.DataFrame(columns=model_columns)
            final_df = pd.concat([final_df, df_encoded], axis=0, ignore_index=True)
            final_df = final_df.fillna(0)
            final_df = final_df[model_columns]

            # Scaling & Prediction
            final_df_imputed = imputer.transform(final_df)
            final_df_scaled = scaler.transform(final_df_imputed)
            
            prediction = model.predict(final_df_scaled)
            output = round(prediction[0], 2)
            
            prediction_text = f"Predicted Sales: {output} €"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
