FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry==2.4.1 && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY . .
RUN poetry install --no-interaction --no-ansi

EXPOSE 8000

# No migration step, unlike the collection services. It does own a database —
# the checkpointer records where each run got to, so a restart resumes an
# analysis instead of starting it over — but LangGraph creates those tables
# itself on startup, so there is nothing to migrate here.
#
# build_graph() runs at import in app/main.py, but it only wires nodes together.
# Every model call this service causes is made by the scoring service behind a
# tool call, so the container starts cleanly with no model credential of its own;
# the only one it reads is the optional judge key, and an absent judge skips the
# judged check rather than failing the run.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
