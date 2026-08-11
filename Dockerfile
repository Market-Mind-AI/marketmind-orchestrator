FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry==2.4.1 && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY . .
RUN poetry install --no-interaction --no-ansi

EXPOSE 8000

# No migration step, unlike the collection services: this one owns no database.
# It holds the LangGraph pipeline and calls the other services over HTTP.
#
# build_graph() runs at import in app/main.py, but it only wires nodes together —
# get_llm() is called inside a node at request time, so the container starts
# cleanly with no API key configured and only fails the individual request that
# needs one.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
