# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
WORKDIR /code
EXPOSE 5001
COPY . .
RUN pip install .
CMD ["flask", "run"]
