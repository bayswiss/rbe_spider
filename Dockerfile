# Image for Binder (https://mybinder.org) and for running the book locally:
#   docker build -t rbe_spider .
#   docker run --rm -it -p 8888:8888 rbe_spider jupyter lab --ip 0.0.0.0
FROM ghcr.io/jorgensd/dolfinx_mpc:v0.12.0.dev1

# The base image has no Jupyter server, which Binder needs
COPY requirements-docs.txt /tmp/requirements-docs.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-docs.txt jupyterlab jupyterhub notebook

# Open the demos, which are python files, as notebooks in JupyterLab
RUN mkdir -p ${VIRTUAL_ENV}/share/jupyter/lab/settings && \
    echo '{"@jupyterlab/docmanager-extension:plugin": {"defaultViewers": {"python": "Jupytext Notebook"}}}' \
    > ${VIRTUAL_ENV}/share/jupyter/lab/settings/overrides.json

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
# Recent Ubuntu releases add `ubuntu` as uid 1000;
# remove it if it already exists before creating our user
RUN id -nu ${NB_UID} && userdel --force $(id -nu ${NB_UID}) || true; \
    useradd -m ${NB_USER} -u ${NB_UID}
ENV HOME=/home/${NB_USER}

# Copy the repository to the home directory for usage in binder
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

USER ${NB_USER}
ENTRYPOINT []
