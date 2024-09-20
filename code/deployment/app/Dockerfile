
FROM python:3.10


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY . /code/app


EXPOSE 8501


CMD ["streamlit", "run", "app/main.py", "--server.port", "8501"]