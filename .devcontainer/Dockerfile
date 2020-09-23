# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.140.1/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version: bionic, focal
ARG VARIANT="focal"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# section to install additional OS packages.

RUN apt-get -q update
RUN apt-get -q install -y curl

RUN curl -fSsL https://storage.googleapis.com/swift-tensorflow-artifacts/releases/v0.11/rc2/swift-tensorflow-RELEASE-0.11-ubuntu20.04.tar.gz -o swift-tensorflow.tar.gz \
    && tar xzf swift-tensorflow.tar.gz --directory / \
    && rm swift-tensorflow.tar.gz

RUN apt-get -q install -y clang
RUN apt-get -q install -y libpython2-dev
RUN apt-get -q install -y libblocksruntime-dev
RUN apt-get -q install -y libxml2
RUN apt-get -q install -y git
RUN apt-get -q install -y zlib1g-dev
RUN apt-get -q install -y python3
RUN apt-get -q install -y python3-lldb

RUN swift --version
