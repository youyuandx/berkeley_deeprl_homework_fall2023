FROM mambaorg/micromamba:latest

# Set up the working directory
WORKDIR /workspaces/your-project-name

# Optional: Copy your environment file and create the environment
# COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
# RUN micromamba create -y -f /tmp/environment.yml && \
#     micromamba clean --all --yes

# Ensure the shell is initialized for the mambauser
RUN micromamba shell init --shell bash --root-prefix=$MAMBA_ROOT_PREFIX