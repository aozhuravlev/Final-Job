import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()
with open('user_action_prediction_pipeline.pkl', 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    session_id: Optional[str] = Field(default=None)
    client_id: Optional[str] = Field(default=None)
    visit_date: Optional[str] = Field(default=None)
    visit_time: Optional[str] = Field(default=None)
    visit_number: Optional[int] = Field(default=None)
    utm_source: Optional[str] = Field(default=None)
    utm_medium: Optional[str] = Field(default=None)
    utm_campaign: Optional[str] = Field(default=None)
    utm_adcontent: Optional[str] = Field(default=None)
    utm_keyword: Optional[str] = Field(default=None)
    device_category: Optional[str] = Field(default=None)
    device_os: Optional[str] = Field(default=None)
    device_brand: Optional[str] = Field(default=None)
    device_model: Optional[str] = Field(default=None)
    device_screen_resolution: Optional[str] = Field(default=None)
    device_browser: Optional[str] = Field(default=None)
    geo_country: Optional[str] = Field(default=None)
    geo_city: Optional[str] = Field(default=None)


class Prediction(BaseModel):
    client_id: str
    # predicted_action: int
    columns: list



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
    cols = [x for x in df.columns]

    return {
        'client_id': form.client_id,
        # 'predicted_action': y[0],
        'columns': cols
    }
