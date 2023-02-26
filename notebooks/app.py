
from fastapi import FastAPI
from typing import Optional
import glob
import os.path
import numpy as np
import pandas as pd
import pickle
import toml
from sklearn.preprocessing import OneHotEncoder
from pydantic import BaseModel,confloat

from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

import joblib

from sklearn.pipeline import Pipeline

def load_pipeline(folder_path):
    # load the pipeline
    #### Write a line that loads the model ..
    model_name = "pipeline.joblib"
    model_path = Path(folder_path, model_name)
    model = joblib.load(model_path)

    # load the feature_columns
    cols_name = "pipeline_cols.joblib"
    cols_path = Path(folder_path, cols_name)
    feature_columns = joblib.load(cols_path)

    return model, feature_columns

app = FastAPI()

class Car(BaseModel):
    year: Optional[float]=2015
    mileage: Optional[float]=77000

    class Config:
        schema_extra = {
            "record": {'year':2015,
                       'mileage':77000,
                      }
        }
        
@app.on_event("startup")
def load_model():
    global pipeline
    global feature_columns

    output_path = Path("pipeline-output")
    pipeline, feature_columns = load_pipeline(output_path)
    
@app.get('/')
def index():
    return {"status": "pass"}


@app.post('/predict')
def get_price_classificatoon(data: Car):
    recieved = data
    print(data)
    year = recieved.year
    mileage = recieved.mileage
    X = np.array([year, mileage]).reshape(1, -1)
    pred_class = pipeline.predict(X)
    pred_proba = pipeline.predict_proba(X)
    if(pred_class==0):
        return {"Class":"Car Price <50k SAR","Probability":round(float(pred_proba[0][0]),3)}
    else:
        return {"Class":"Car Price >=50k SAR","Probability":round(float(pred_proba[0][1]),3)}
