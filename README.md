<img src="figures/slide.png" alt="Alt text" width="700">

# multiNodeGpus_example
Example for running multi-Gpus training across multiple nodes on the MPCDF cluster

## Installation

```bash
git clone https://github.com/mvigl/multiNodeGpus_example.git

cd multiNodeGpus_example

pip install -e .

```

## Usage
From the `MPCDF` cluster, `raven (NVIDIA)` or `viper (AMD)`, after chnaging the paths in the slurm config from `/raven(viper)/u/mvigl/multiNodeGpus_example` to `/raven(viper)/u/<user>/path/to/multiNodeGpus_example`, run:

```bash
bash slurm/submit_jobs(_raven).sh
```