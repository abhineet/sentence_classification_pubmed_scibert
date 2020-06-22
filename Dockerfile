FROM python:3.7

WORKDIR /src/

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN wget  "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar" --directory-prefix=../data/ && tar -xf data/scibert_scivocab_uncased.tar -C ../data/
RUN tar -xf data/train.tar.xz -C ../data/