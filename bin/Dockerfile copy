# vim: ft=mako

FROM singularitybase.azurecr.io/base/job/pytorch/acpt-torch2.5.0-py3.10-cuda12.4-ubuntu22.04:20241022T103546977 as base

FROM base

RUN which mamba || conda install -yq -c conda-forge mamba
RUN mamba install -yq -c biobuilds -c bioconda -c pytorch -c huggingface -c nvidia -c anaconda -c conda-forge -c defaults python=3.12.4 pytorch=2.5.1 pip=24.2 pandas=1.5.2 wandb=0.18.7 tqdm=4.66.5 pytables=3.10.1 huggingface_hub=0.26.2 transformers=4.45.2 python-dotenv=1.0.1 matplotlib=3.9.1  && conda clean -yqa
RUN python -mpip install -q  torcheval==0.0.7

# set validation arguments for expected use of this image
ENV SINGULARITY_IMAGE_FRAMEWORK=pytorch
ENV SINGULARITY_IMAGE_ACCELERATOR=NVIDIA