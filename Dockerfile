FROM python:3.10-slim

ARG BUILD_DATE
ARG VCS_REF

LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.name="clinical-care-rl-comm"
LABEL org.label-schema.description="Healthcare Communication Optimization using Reinforcement Learning"
LABEL org.label-schema.vcs-url="https://github.com/okahwaji-tech/clinical-care-rl-comm"
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.vendor="Omar Kahwaji"

# Override default SIGTERM to SIGINT (Ctrl+C)
STOPSIGNAL SIGINT

ENV DEBIAN_FRONTEND="noninteractive" \
    PYTHONPATH=/app \
    SOURCEDIR=/app \
    COMMIT_HASH=$VCS_REF

RUN mkdir -p ${SOURCEDIR}
WORKDIR ${SOURCEDIR}

RUN apt-get update -yy && \
    apt-get install -yy --no-install-recommends gcc libc-dev && \
    apt-get -y --purge autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy project files
COPY . .

# Install Poetry and dependencies
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main

# Make scripts executable
RUN chmod +x entrypoint.sh

# Create non-root user
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser ${SOURCEDIR}

USER appuser

ENTRYPOINT [ "./entrypoint.sh" ]
CMD [ "python", "scripts/train.py", "--help" ]
