FROM python:3.10.9

WORKDIR /root/projects

RUN pip3 install --upgrade pip
RUN pip3 install ipykernel
RUN pip3 install braindecode
RUN pip3 install torch
RUN pip3 install neptune neptune-sklearn
RUN pip3 install wandb
RUN pip3 install moabb

CMD tail -f /dev/null