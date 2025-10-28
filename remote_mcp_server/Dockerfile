FROM python:3.10

# Setting time
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV TZ Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y && apt-get install -y \
    locales-all \
    tzdata && rm -rf /var/lib/apt/lists/*

# TINI
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# user setting
RUN useradd -U -s /usr/sbin/nologin -u 2002 -m mongttang

USER mongttang
WORKDIR /home/mongttang/app

ENV PATH="/home/mongttang/.local/bin:${PATH}"
RUN pip install --upgrade pip

# env
# disable stdout buffering
ENV PYTHONUNBUFFERED=1

ARG HUBBLE_API_KEY
ENV HUBBLE_API_KEY $HUBBLE_API_KEY

# requirements.txt
COPY --chown=mongttang:mongttang requirements.txt /home/mongttang/app
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=mongttang:mongttang ./src/ /home/mongttang/app/

CMD ["python3", "main.py"]