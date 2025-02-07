FROM python:3.9
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 8501
RUN chmod +x ./heroku_startup.sh
ENTRYPOINT "./heroku_startup.sh" 