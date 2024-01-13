FROM python:3.11-slim-buster

# Install build dependencies
RUN apt-get update -y && apt-get install -y build-essential

WORKDIR /app
COPY . /app

# Install AWS CLI
RUN apt-get install awscli -y

# Install Python dependencies
RUN pip install -r requirements.txt

CMD ["python3", "application.py"]
