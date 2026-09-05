# A reviewer should be able to get from `git clone` to a running API without installing a
# Python toolchain or discovering which of their four interpreters has the right wheels.
#
# Two stages. The first installs dependencies against a pinned base; the second copies the
# resulting site-packages and the source. The split exists so that editing a source file
# does not re-resolve numpy, scipy and scikit-learn, which is most of the build time.
FROM python:3.11-slim-bookworm AS deps

# Compilers are not needed: every pinned dependency ships a manylinux wheel for this base.
# If that ever stops being true the build will fail loudly here rather than silently
# producing a slower interpreter-compiled extension.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt requirements-serve.txt ./
RUN pip install --prefix=/install -r requirements-serve.txt


FROM python:3.11-slim-bookworm AS runtime

# Unbuffered so the stage-by-stage progress of a run reaches `docker logs` while the run is
# still going. A pipeline that prints nothing for four minutes looks hung.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    # Each estimator in the ensemble is already parallel over trees. Letting BLAS also fan
    # out oversubscribes the container's CPU quota and makes the run slower, not faster.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY --from=deps /install /usr/local

WORKDIR /app
COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml README.md LICENSE ./

# Non-root. The service reads a model bundle and writes a SQLite file; nothing it does needs
# to be able to write to /usr, and an unauthenticated HTTP surface is exactly the process
# you do not want running as root.
RUN useradd --create-home --uid 10001 redteam \
    && mkdir -p /app/artifacts /app/data \
    && chown -R redteam:redteam /app
USER redteam

EXPOSE 8000

# Reports degraded rather than dead when no bundle is present, so `docker compose up` on a
# fresh clone comes up healthy and the demo target can then populate it.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# 0.0.0.0 inside the container only. The compose file publishes it on loopback, because
# this API has no authentication and binding it to a LAN interface would hand anyone on the
# network a free scoring oracle.
CMD ["python", "-m", "uvicorn", "--factory", "redteam.serve.api:get_app", \
     "--host", "0.0.0.0", "--port", "8000"]
