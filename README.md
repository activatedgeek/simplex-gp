# Bilateral GPs

## Setup

Create `conda` environment

```shell
$ conda env create
```

Remember to add the root of the project to PYTHONPATH if not already.

```shell
$ export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

### Datasets

Download the `mat` files for appropriate UCI datasets from [here](https://cims.nyu.edu/~andrewgw/pattern/).

## Run

```
$ python experiments/train.py --dataset=elevators --data-dir=<path/to/uci/data>
```

The CLI uses Fire, so all function arguments can be used as CLI arguments.

To disable CUDA (and avoid segfaults for now), `export CUDA_VISIBLE_DEVICES=-1`.

### W&B

See [Weights & Biases Configuration Docs](https://docs.wandb.com/sweeps/configuration)
for all options. First, setup environment variables required

```shell
export WANDB_API_KEY=<api_key> # https://app.wandb.ai/settings (not needed for local runs)
export WANDB_DISABLE_CODE=true # optional
export WANDB_MODE=run # to upload results to W&B account
```

Then, create a new sweep. [configs](./configs) folder contains sweep files for
various experiments.

```shell
> wandb sweep </path/to/config.yml>
```

Finally, run the agent using

```shell
> wandb agent --count=1 <sweep-id>
```

**NOTE**: Use `--count=1` so as to only run one run per program call. We don't
want memory pollution in the same process run. Run multiple independent jobs
instead.
