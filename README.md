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

## Run

Download the `mat` files for appropriate UCI dataset and then,

```
$ python experiments/train.py --dataset=elevators --data-dir=<path/to/uci/data>
```

The CLI uses Fire, so all function arguments can be used as CLI arguments.

To disable CUDA (and avoid segfaults for now), `export CUDA_VISIBLE_DEVICES=-1`.
