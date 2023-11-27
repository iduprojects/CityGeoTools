FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-2021-10-02

RUN apt-get update && apt-get -qq -y install curl \
    && apt-get update && apt-get -y install libspatialindex-dev \
    && apt-get clean

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

CMD uvicorn app.main:app --host 0.0.0.0 --port 5000
