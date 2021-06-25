FROM python:3.9.1-slim-buster

WORKDIR /app

COPY requirements.txt /app
# RUN python3.9 -m venv venv
# RUN source venv/bin/activate
RUN python3 --version
RUN pip --version
RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["python3.9"] 
CMD ["main.py"]