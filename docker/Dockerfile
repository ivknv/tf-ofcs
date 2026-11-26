FROM rocm/tensorflow:rocm6.1-py3.10-tf2.15-runtime

RUN mkdir -p /tf-ofcs
WORKDIR /tf-ofcs

RUN apt -y update &&\
    apt -y upgrade &&\
    apt -y install python3-plotly &&\
    pip install silence_tensorflow numba

ENV PYTHONPATH="/tf-ofcs/src:${PYTHONPATH}"

ENTRYPOINT ["python3", "-m", "tf_ofcs"]
