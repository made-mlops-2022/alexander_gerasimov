FROM python:3.9.6

COPY train.py /train.py
COPY requirements.txt /requirements.txt
RUN pip install --no-cache -r requirements.txt

ENTRYPOINT ["python", "train.py"]