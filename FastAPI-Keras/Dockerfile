FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install tensorflow==2.4.1

COPY ./model/tf_keras_imdb /model/

COPY ./app /app
