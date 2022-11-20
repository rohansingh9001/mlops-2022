FROM python:3.8-slim

WORKDIR /usr/src/lab2

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3", "classification.py"]
