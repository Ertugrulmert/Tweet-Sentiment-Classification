FROM conda/miniconda3
WORKDIR /app

# install node
ENV NODE_VERSION=16.13.0
RUN apt-get update; apt-get install -y curl
RUN apt-get install  -y libgomp1
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"

# prepare python env
RUN conda install pytorch=0.4.1 -c pytorch

# run
ENTRYPOINT [ "./script.sh" ]