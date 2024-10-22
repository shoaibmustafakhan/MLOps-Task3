from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

app = FastAPI()
model = tf.keras.models.load_model('model.h5')

class PredictionInput(BaseModel):
    input: list

@app.post('/predict')
def predict(data: PredictionInput):
    prediction = model.predict(np.array([data.input]))
    return {"prediction": prediction.tolist()}
