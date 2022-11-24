FROM python:3.9.6
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY src /src
COPY app.py /app.py
COPY model.pkl /model.pkl

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]