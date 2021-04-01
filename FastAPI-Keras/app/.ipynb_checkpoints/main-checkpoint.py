import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
import tensorflow as tf
app = FastAPI()

model = tf.keras.models.load_model('model/tf_keras_imdb')


class Reviews(BaseModel):
    """[summary]

    Args:
        BaseModel ([type]): [description]
    """
    review: str


@app.get('/')
def index():
    return {'message': 'This is IMDb Reviews Classification API!'}


@app.post('/predict')
def predict_review(data: Reviews):
    """[summary]

    Args:
        data (Reviews): [description]

    Returns:
        [type]: [description]
    """
    data = data.dict()
    prediction = model.predict([data['review']])
    return {
        'prediction': prediction.tolist()[0][0]
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)
