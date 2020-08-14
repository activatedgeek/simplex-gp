import os
import torch
from torch.utils.tensorboard import SummaryWriter
import gpytorch
from tqdm.auto import tqdm
import wandb

from bi_gp.bilateral_kernel import BilateralKernel
from utils import UCIDataset


def set_seeds(seed=None):
  if seed:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(BilateralKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main(dataset, epochs=100, lr=0.1, log_int=10, seed=None):
    set_seeds(seed)

    wandb.init(tensorboard=True)

    device = "gpu" if torch.cuda.is_available() else "cpu"
    logger = SummaryWriter(log_dir=wandb.run.dir)

    train_dataset = UCIDataset.create(dataset, mode="train", device=device)
    test_dataset = UCIDataset.create(dataset, mode="test", device=device)

    train_x, train_y = train_dataset.x, train_dataset.y
    test_x, test_y = test_dataset.x, test_dataset.y

    x_mean = train_x.mean(0)
    x_std = train_x.std(0) + 1e-6

    y_mean = train_y.mean(0)
    y_std = train_y.std(0) + 1e-6

    train_x = (train_x - x_mean) / x_std
    train_y = (train_y - y_mean) / y_std

    test_x = (test_x - x_mean) / x_std
    test_y = (test_y - y_mean) / y_std

    model = ExactGPModel(train_x, train_y).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    for i in tqdm(range(epochs)):
        model.train()
        
        optimizer.zero_grad()

        output = model(train_x)

        loss = -mll(output, train_y)
        loss.backward()

        optimizer.step()

        if i % log_int == 0:
            logger.add_scalar('train/loss', loss.detach().item(), global_step=i + 1)

            model.eval()
            with torch.no_grad():
                pred_y = likelihood(model(test_x)).mean
                rmse = (pred_y - test_y).pow(2).mean(0).sqrt()
                logger.add_scalar('test/rmse', rmse.item(), global_step=i + 1)


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import fire
  fire.Fire(main)
