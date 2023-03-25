# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry install

# Generate requirements.txt
RUN poetry export --without-hashes --format=requirements.txt > requirements.txt

# Install python packages (banana deps: sanic==22.6.2, accelerate)
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Download model weights
ADD download.py .
RUN python3 download.py


# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py
