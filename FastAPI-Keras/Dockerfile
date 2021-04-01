FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install tensorflow==2.4.1

COPY ./model /model/

COPY ./app /app

EXPOSE 8010

CMD ["python", "main.py"]