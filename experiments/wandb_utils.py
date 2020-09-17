import wandb
import os
from datetime import datetime
from tqdm.auto import tqdm


def download_sweep_events(s_id=None, path=None):
  assert s_id is not None
  assert path is not None

  api = wandb.Api()
  sweep = api.sweep(s_id)
  for run in sweep.runs:
    events_file = [f for f in run.files() if 'events' in f.name][0]
    run_path = os.path.join(path, run.path[1], run.name)
    os.makedirs(run_path, exist_ok=True)
    events_file.download(root=run_path, replace=True)


def generate_sweep(alg=None, dataset=None, lr=0.05, lanc_iter=50, pre_size=10,
                   project='bilateral-gp-experiments', submit=False):
  assert alg in ['bigp', 'skip']
  assert dataset is not None

  md = datetime.now().strftime('%h%d')
  sweep_config = {
    'name': f'[{md}] {alg}-{dataset}-{lr}-{lanc_iter}-{pre_size}',
    'method': 'random',
    'parameters': {
      'dataset': {
        'value': dataset
      },
      'lr': {
        'value': lr
      },
      'epochs': {
        'value': 100
      },
      'lanc_iter': {
        'value': lanc_iter
      },
      'pre_size': {
        'value': pre_size
      }
    },
    'program': f'experiments/train_{alg}.py',
    'command': [
      '${env}',
      '${interpreter}',
      '${program}',
      '${args}',
    ]
  }
  
  if submit:
    sweep_id = wandb.sweep(sweep_config, project=project)
    return sweep_id
  
  return sweep_config


def generate_sweep_all(alg=None, submit=False):
  ds = [
    '3droad',
    'elevators',
    'houseelectric',
    'pumadyn32nm_all',
    'protein',
    'kin40k_all',
    'keggdirected',
    'precipitation_all'
  ]

  sweep_ids = []
  for dataset in tqdm(ds):
    s_id = generate_sweep(alg=alg, dataset=dataset, submit=submit)
    sweep_ids.append(s_id)

  if submit:
    print('\n'.join(sweep_ids))


if __name__ == "__main__":
  import fire
  fire.Fire(dict(
    dse=download_sweep_events,
    gs=generate_sweep,
    gsa=generate_sweep_all,
  ))
