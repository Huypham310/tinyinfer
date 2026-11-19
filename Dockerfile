# Build + test image for TinyInfer. Same toolchain the numbers in the README were measured with
# (Debian bookworm = GCC 12), so "works in the container" means the same as "works in CI".
#
#   docker build -t tinyinfer .
#   docker run --rm tinyinfer                       # coverage build, ctest, lcov gate, pytest
#   docker run --rm tinyinfer ctest --test-dir /tmp/build --output-on-failure   (after a build)
#
# Not a runtime image: it is a throwaway environment to build and check the engine.
FROM python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# python:3.12-bookworm already ships gcc/g++ 12; cmake 3.25 from apt satisfies our 3.24 minimum
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake ninja-build lcov \
    && rm -rf /var/lib/apt/lists/*

# test deps in their own layer so editing C++ doesn't re-download them
RUN pip install --no-cache-dir pytest numpy

WORKDIR /src
COPY . /src

# stays root on purpose: it is a throwaway container and check_linux.sh can apt-install
# anything missing when it runs as root
# TODO: multi-stage build if we ever ship a slim CLI image (khong can luc nay)

# default: the full gate (coverage >= MIN_COVERAGE, then pytest). Override MIN_COVERAGE / SKIP_PYTHON
# with -e on `docker run`
CMD ["bash", "scripts/check_linux.sh"]
