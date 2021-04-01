FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN pip install joblib scikit-learn

COPY ./model /model/

COPY ./app /app
