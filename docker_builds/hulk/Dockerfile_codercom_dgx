FROM python:3.9-slim-buster

RUN apt-get -y update && apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get -y update && apt-get install -y --no-install-recommends \
        wget \
	python3-setuptools \
	python3-sphinx \
	nodejs \
	gcc \
	git \
	vim \
	make \
	ffmpeg \
	libsm6 \
	sudo \
	python3-pip \
        && apt-get clean all && rm -rf /var/lib/apt/lists/*

WORKDIR cocpit
RUN pip3 install --upgrade pip
COPY requirements_dev.txt requirements_dev.txt
RUN pip3 install -r requirements_dev.txt

# install cocpit package for dev
COPY setup.py setup.py
COPY README.md README.md
RUN pip3 install -e .

WORKDIR /usr/local/bin
RUN wget https://github.com/cdr/code-server/releases/download/v4.7.1/code-server-4.7.1-linux-amd64.tar.gz  && \
	tar -xzvf code-server-4.7.1-linux-amd64.tar.gz
ENV PATH="/usr/local/bin/code-server-4.7.1-linux-amd64/bin:$PATH"
