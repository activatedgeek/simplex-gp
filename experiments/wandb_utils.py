import wandb
import os


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


if __name__ == "__main__":
  import fire
  fire.Fire(download_sweep_events)
