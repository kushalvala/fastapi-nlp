import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel
app = FastAPI()
model = joblib.load('model_pipeline.joblib')

class Stroke(BaseModel):
    gender : str
    age : int
    hypertension : int
    heart_disease : int
    ever_married : str
    work_type : str
    Residence_type : str
    avg_glucose_level : float
    bmi : float 
    smoking_status : str
        
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/predict')
def predict_stroke(data : Stroke):
    data = data.dict()
    gender = data['gender']
    age = data['age']
    hypertension = data['hypertension']
    heart_disease = data['heart_disease']
    ever_married = data['ever_married']
    work_type = data['work_type']
    Residence_type = data['Residence_type']
    avg_glucose_level = data['avg_glucose_level']
    bmi = data['bmi']
    smoking_status = data['smoking_status']
    
    prediction = model.predict([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi, smoking_status]])
    
    if (prediction[0] == 0):
        prediction = 'No Stroke'
    else:
        prediction = 'Stroke'
        
    return {
        'prediction' : prediction
        }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    