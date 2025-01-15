FROM python:3.11-slim

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./



COPY ./package[s] ./packages

COPY ./state_of_the_union.txt /code/state_of_the_union.txt

COPY ./Manuscripts_Speaking_The_History_of_Read.pdf /code/Manuscripts_Speaking_The_History_of_Read.pdf

RUN poetry install  --no-interaction --no-ansi --no-root

RUN pip install --upgrade cryptography


COPY ./app ./app

COPY ./.env /code/.env

RUN poetry install --no-interaction --no-ansi


RUN pip install --no-cache-dir "pydantic>=2.5.3" "pydantic-core>=2.14.6"


EXPOSE 8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]