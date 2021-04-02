import uvicorn
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
import tensorflow as tf
app = FastAPI()

model = tf.keras.models.load_model('../model/tf_keras_imdb')


class Reviews(BaseModel):
    review: str


@app.get('/')
def index():
    return {'message': 'This is IMDb Reviews Classification API!'}


@app.post('/predict')
def predict_review(data: Reviews):
    """ FastAPI 

    Args:
        data (Reviews): json file 

    Returns:
        prediction: probability of review being positive
    """
    data = data.dict()
    review = data['review']
    prediction = model.predict([review])
    return {
        'prediction': prediction.tolist()[0][0]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
