FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12.py310

RUN pip install tensorflow-hub
RUN pip install tf-models-official==2.13.2

COPY . /code

WORKDIR /code

ENTRYPOINT ["python3", "-m", "trainer.task"]
