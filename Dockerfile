FROM pytorch/pytorch:latest
MAINTAINER Vanessa Przybylo vprzybylo@albany.edu
WORKDIR /data/data
RUN apt-get -y update && apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get -y update && apt-get install -y nodejs gcc git vim ffmpeg libsm6 sudo
RUN groupadd -o -g 1001 vanessa \
        && useradd -o -r -m -u 1001 -g 1001 vanessa
COPY requirements_new.txt requirements_new.txt
RUN pip install -r requirements_new.txt
 
