ARG VARIANT=gpu  # Default to GPU if not provided
FROM thewillyp/devenv:master-1.0.32-${VARIANT}

WORKDIR /

RUN git clone https://github.com/thewillyP/rnn-test.git

WORKDIR /rnn-test

RUN chmod -R 777 /rnn-test

RUN pip install .

RUN mkdir -p /wandb_data

RUN chmod -R 777 /wandb_data

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]