FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools


# Install the requirements
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user -r requirements.txt

# Download the model, tokenizer and metrics
RUN mkdir -p /opt/app/models
COPY --chown=user:user download_model.py /opt/app/
RUN python download_model.py --model_name CLTL/MedRoBERTa.nl
COPY --chown=user:user download_metrics.py /opt/app/
RUN python download_metrics.py

# Set the environment variables
ENV TRANSFORMERS_OFFLINE=1
ENV HF_EVALUATE_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Copy the algorithm code
COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
