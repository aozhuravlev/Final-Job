import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()
with open('user_action_prediction_pipeline.pkl', 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    session_id: Optional[str]
    client_id: Optional[str]
    visit_date: Optional[str]
    visit_time: Optional[str]
    visit_number: Optional[int]
    utm_source: Optional[str]
    utm_medium: Optional[str]
    utm_campaign: Optional[str]
    utm_adcontent: Optional[str]
    utm_keyword: Optional[str]
    device_category: Optional[str]
    device_os: Optional[str]
    device_brand: Optional[str]
    device_model: Optional[str]
    device_screen_resolution: Optional[str]
    device_browser: Optional[str]
    geo_country: Optional[str]
    geo_city: Optional[str]


class Prediction(BaseModel):
    client_id: str
    predicted_action: int
    
    

@app.get('/status')
def status():
    return "I await your commands, my Lord."


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.model_dump()])
    y = model['model'].predict(df)
    
    
    return {
        'client_id': form.client_id,
        'predicted_action': y[0],
    }
