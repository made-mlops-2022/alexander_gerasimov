FROM python:3.9.6

COPY download.py /download.py
COPY requirements.txt /requirements.txt
RUN pip install --no-cache -r requirements.txt

ENTRYPOINT ["python", "download.py"]