from dataclasses import dataclass
import random
import os, time
from typing import Union, Literal

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
from metaopt.mnist.mlp import MLP
from metaopt.util import to_torch_variable
from metaopt.util_ml import compute_correlation
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sweep_agent.agent import get_sweep_config
import wandb
import joblib

TRAIN = 0
VALID = 1
TEST = 2


@dataclass
class Config:
    meta_optimizer: Literal["sgd", "adam"] = "sgd"
    use_64: int = 0
    hv_r: float = 1e-3
    dataset: Literal["mnist", "fashionmnist"] = "mnist"
    project: str = "new_metaopt"
    test_freq: int = 10
    rng: int = 0
    num_epoch: int = 100
    batch_size: int = 100
    batch_size_vl: int = 100
    model_type: str = "rnn"
    opt_type: str = "sgd"
    xdim: int = 784
    hdim: int = 128
    ydim: int = 10
    num_hlayers: int = 3
    lr: float = 1e-3
    mlr: float = 1e-4
    lambda_l2: float = 1e-4
    update_freq: int = 1
    reset_freq: int = 0
    valid_size: int = 10000
    checkpoint_freq: int = 10
    is_cuda: int = 0
    save: int = 0
    save_dir: str = "/scratch"


class VanillaRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr_init, lambda_l2, is_cuda):
        super(VanillaRNNModel, self).__init__()
        input_size = int(input_size)
        print(input_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.eta = lr_init
        self.lambda_l2 = lambda_l2
        self.is_cuda = is_cuda
        self.name = "VanillaRNN"
        print(input_size)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        if is_cuda:
            self.rnn = self.rnn.cuda()
            self.fc = self.fc.cuda()

        self.n_params = sum(p.numel() for p in self.parameters())
        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)

        self.reset_jacob(is_cuda)

    def reset_jacob(self, is_cuda):
        self.dFdlr = torch.zeros(self.n_params)
        self.dFdl2 = torch.zeros(self.n_params)
        self.dFdl2_norm = 0
        self.dFdlr_norm = 0
        if is_cuda:
            self.dFdlr = self.dFdlr.cuda()
            self.dFdl2 = self.dFdl2.cuda()

    def forward(self, x, logsoftmaxF=1):
        if x.dim() > 2:
            x = x.view(x.size(0), -1, self.input_size)

        # Create random normal hidden state
        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)  # (num_layers, batch_size, hidden_size)

        # if self.is_cuda:
        #     h0 = h0.cuda()

        # out, _ = self.rnn(x, h0)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1) if logsoftmaxF else F.softmax(out, dim=1)

    def update_dFdlr(self, Hv, param, grad):
        alpha = F.softplus(torch.tensor(self.eta))
        lambd = F.softplus(torch.tensor(self.lambda_l2))
        sigmoid_alpha = torch.sigmoid(torch.tensor(self.eta))

        grad_term = grad + 2 * lambd * param
        self.Hlr_norm = norm(alpha * Hv)
        self.dFdlr_norm = norm(self.dFdlr)
        self.dFdlr.data = self.dFdlr.data * (1 - 2 * lambd * alpha) - alpha * Hv - grad_term * sigmoid_alpha

    def update_dFdlambda_l2(self, Hv, param):
        alpha = F.softplus(torch.tensor(self.eta))
        lambd = F.softplus(torch.tensor(self.lambda_l2))
        sigmoid_lambda = torch.sigmoid(torch.tensor(self.lambda_l2))

        self.Hl2 = alpha * Hv
        self.Hl2_norm = torch.norm(self.Hl2)
        self.dFdl2_norm = torch.norm(self.dFdl2)

        # Multiply ONLY the last term by sigmoid_lambda
        self.dFdl2.data = self.dFdl2.data * (1 - 2 * lambd * alpha) - self.Hl2 - (2 * alpha * param) * sigmoid_lambda

    # def update_eta(self, mlr, val_grad):
    #     delta = val_grad.dot(self.dFdlr).data.cpu().numpy()
    #     self.eta -= mlr * delta
    #     self.eta = max(0.0, self.eta)
    #
    # def update_lambda(self, mlr, val_grad):
    #     delta = val_grad.dot(self.dFdl2).data.cpu().numpy()
    #     self.lambda_l2 -= mlr * delta
    #     self.lambda_l2 = np.maximum(0, self.lambda_l2)
    #     # self.lambda_l2 = np.clip(self.lambda_l2, 0, 0.0002)

    def update_hyperparams(self, delta_eta, delta_lambda, mlr=1e-3, method="adam", beta1=0.9, beta2=0.999, eps=1e-8):
        if not hasattr(self, "t"):  # Time step
            self.t = 0
            self.m_eta = 0
            self.v_eta = 0
            self.m_lambda = 0
            self.v_lambda = 0

        self.t += 1  # increment step

        if method == "sgd":
            self.eta -= mlr * delta_eta
            self.lambda_l2 -= mlr * delta_lambda

        elif method == "adam":
            # ETA
            self.m_eta = beta1 * self.m_eta + (1 - beta1) * delta_eta
            self.v_eta = beta2 * self.v_eta + (1 - beta2) * (delta_eta**2)
            m_hat_eta = self.m_eta / (1 - beta1**self.t)
            v_hat_eta = self.v_eta / (1 - beta2**self.t)
            self.eta -= mlr * m_hat_eta / (np.sqrt(v_hat_eta) + eps)

            # LAMBDA
            self.m_lambda = beta1 * self.m_lambda + (1 - beta1) * delta_lambda
            self.v_lambda = beta2 * self.v_lambda + (1 - beta2) * (delta_lambda**2)
            m_hat_lambda = self.m_lambda / (1 - beta1**self.t)
            v_hat_lambda = self.v_lambda / (1 - beta2**self.t)
            self.lambda_l2 -= mlr * m_hat_lambda / (np.sqrt(v_hat_lambda) + eps)


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr_init, lambda_l2, is_cuda):
        super(RNNModel, self).__init__()
        input_size = int(input_size)
        print(input_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.eta = lr_init  # For consistency with MLP code
        self.lambda_l2 = lambda_l2
        self.is_cuda = is_cuda
        self.name = "RNN"

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        if is_cuda:
            self.lstm = self.lstm.cuda()
            self.fc = self.fc.cuda()

        self.n_params = sum(p.numel() for p in self.parameters())
        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)

        self.reset_jacob(is_cuda)

    def reset_jacob(self, is_cuda):
        self.dFdlr = torch.zeros(self.n_params)
        self.dFdl2 = torch.zeros(self.n_params)
        self.dFdl2_norm = 0
        self.dFdlr_norm = 0
        if is_cuda:
            self.dFdlr = self.dFdlr.cuda()
            self.dFdl2 = self.dFdl2.cuda()

    def forward(self, x, logsoftmaxF=1):
        # Automatically reshape if input is flat (e.g., B x 784)
        if x.dim() > 2:
            new_shape = (x.size(0), -1, self.input_size)  # Batch size is fixed, others are flattened
            x = x.view(new_shape)

        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Last time step

        if logsoftmaxF:
            return F.log_softmax(out, dim=1)
        else:
            return F.softmax(out, dim=1)

    def update_dFdlr(self, Hv, param, grad):
        alpha = F.softplus(torch.tensor(self.eta))
        lambd = F.softplus(torch.tensor(self.lambda_l2))
        sigmoid_alpha = torch.sigmoid(torch.tensor(self.eta))

        grad_term = grad + 2 * lambd * param
        self.Hlr_norm = torch.norm(alpha * Hv)
        self.dFdlr_norm = torch.norm(self.dFdlr)
        self.dFdlr.data = self.dFdlr.data * (1 - 2 * lambd * alpha) - alpha * Hv - grad_term * sigmoid_alpha

    def update_dFdlambda_l2(self, Hv, param):
        alpha = F.softplus(torch.tensor(self.eta))
        lambd = F.softplus(torch.tensor(self.lambda_l2))
        sigmoid_lambda = torch.sigmoid(torch.tensor(self.lambda_l2))

        self.Hl2 = alpha * Hv
        self.Hl2_norm = torch.norm(self.Hl2)
        self.dFdl2_norm = torch.norm(self.dFdl2)

        # Multiply ONLY the last term by sigmoid_lambda
        self.dFdl2.data = self.dFdl2.data * (1 - 2 * lambd * alpha) - self.Hl2 - (2 * alpha * param) * sigmoid_lambda

    # def update_eta(self, mlr, val_grad):
    #     delta = val_grad.dot(self.dFdlr).data.cpu().numpy()
    #     self.eta -= mlr * delta
    #     self.eta = max(0.0, self.eta)
    #
    # def update_lambda(self, mlr, val_grad):
    #     delta = val_grad.dot(self.dFdl2).data.cpu().numpy()
    #     self.lambda_l2 -= mlr * delta
    #     self.lambda_l2 = np.maximum(0, self.lambda_l2)
    #     # self.lambda_l2 = np.clip(self.lambda_l2, 0, 0.0002)

    def update_hyperparams(self, delta_eta, delta_lambda, mlr=1e-3, method="adam", beta1=0.9, beta2=0.999, eps=1e-8):
        if not hasattr(self, "t"):  # Time step
            self.t = 0
            self.m_eta = 0
            self.v_eta = 0
            self.m_lambda = 0
            self.v_lambda = 0

        self.t += 1  # increment step

        if method == "sgd":
            self.eta -= mlr * delta_eta
            self.lambda_l2 -= mlr * delta_lambda

        elif method == "adam":
            # ETA
            self.m_eta = beta1 * self.m_eta + (1 - beta1) * delta_eta
            self.v_eta = beta2 * self.v_eta + (1 - beta2) * (delta_eta**2)
            m_hat_eta = self.m_eta / (1 - beta1**self.t)
            v_hat_eta = self.v_eta / (1 - beta2**self.t)
            self.eta -= mlr * m_hat_eta / (np.sqrt(v_hat_eta) + eps)

            # LAMBDA
            self.m_lambda = beta1 * self.m_lambda + (1 - beta1) * delta_lambda
            self.v_lambda = beta2 * self.v_lambda + (1 - beta2) * (delta_lambda**2)
            m_hat_lambda = self.m_lambda / (1 - beta1**self.t)
            v_hat_lambda = self.v_lambda / (1 - beta2**self.t)
            self.lambda_l2 -= mlr * m_hat_lambda / (np.sqrt(v_hat_lambda) + eps)


def save_object_as_wandb_artifact(obj, artifact_name: str, fdir: str, filename: str, artifact_type: str) -> None:
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"

    full_path = os.path.join(fdir, filename)
    joblib.dump(obj, full_path, compress=0)
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(full_path)
    wandb.log_artifact(artifact)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_dataset(args, dataset_fn):
    test_dataset = dataset_fn(
        f"{args.save_dir}/data/mnist",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    dataset = dataset_fn(
        f"{args.save_dir}/data/mnist", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )
    train_set, valid_set = torch.utils.data.random_split(dataset, [60000 - args.valid_size, args.valid_size])

    data_loader_tr = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, num_workers=0
    )
    data_loader_vl = DataLoader(
        valid_set, batch_size=args.batch_size_vl, shuffle=True, worker_init_fn=seed_worker, num_workers=0
    )
    data_loader_te = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        num_workers=0,
    )

    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    data_loader_vl = infinite_loader(data_loader_vl)
    dataset = [data_loader_tr, data_loader_vl, data_loader_te]
    return dataset


def softplus_inverse(x):
    """
    Computes the inverse softplus using numpy:
    softplus_inverse(x) = log(exp(x) - 1)

    Uses a numerically stable form:
      softplus_inverse(x) = x + log(1 - exp(-x))

    Args:
      x: np.ndarray or float, input values (assumed non-negative).

    Returns:
      np.ndarray or float, same shape as input.
    """
    x = np.asarray(x)

    eps = np.finfo(x.dtype).eps
    threshold = np.log(eps) + 2.0

    is_too_small = x < np.exp(threshold)
    is_too_large = x > -threshold

    too_small_value = np.log(x)
    too_large_value = x

    # Safe version of x for computing inverse softplus
    safe_x = np.where(is_too_small | is_too_large, 1.0, x)

    y = safe_x + np.log1p(-np.exp(-safe_x))  # log(1 - exp(-x)) using log1p for stability

    # Compose the output based on conditions
    result = np.where(is_too_small, too_small_value, np.where(is_too_large, too_large_value, y))
    return result


def main(args: Config):
    args.lr = float(softplus_inverse(args.lr))
    args.lambda_l2 = float(softplus_inverse(args.lambda_l2))

    match args.dataset:
        case "mnist":
            dataset = load_dataset(args, datasets.MNIST)
        case "fashionmnist":
            dataset = load_dataset(args, datasets.FashionMNIST)
        case _:
            raise NotImplementedError

    ## Initialize Model and Optimizer
    hdims = [args.xdim] + [args.hdim] * args.num_hlayers + [args.ydim]
    num_layers = args.num_hlayers + 2
    num_layers = int(num_layers)
    hdims = [int(x) for x in hdims]

    match args.model_type:
        case "rnn":
            model = VanillaRNNModel(
                input_size=hdims[0],
                hidden_size=hdims[1],
                num_layers=num_layers - 2,
                output_size=hdims[-1],
                lr_init=args.lr,
                lambda_l2=args.lambda_l2,
                is_cuda=args.is_cuda,
            )
        case "lstm":
            model = RNNModel(
                input_size=hdims[0],
                hidden_size=hdims[1],
                num_layers=num_layers - 2,
                output_size=hdims[-1],
                lr_init=args.lr,
                lambda_l2=args.lambda_l2,
                is_cuda=args.is_cuda,
            )
        case "mlp":
            model = MLP(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=args.is_cuda)
        case _:
            raise ValueError("Invalid model type. Choose 'mlp', 'rnn', or 'lstm'.")

    optimizer = optim.SGD(
        model.parameters(), lr=F.softplus(torch.tensor(args.lr)), weight_decay=F.softplus(torch.tensor(args.lambda_l2))
    )

    print(
        "Model Type: %s Opt Type: %s Update Freq %d Reset Freq %d"
        % (args.model_type, args.opt_type, args.update_freq, args.reset_freq)
    )

    os.makedirs("%s/exp/mnist/" % args.save_dir, exist_ok=True)
    os.makedirs("%s/exp/mnist/mlr%f_lr%f_l2%f/" % (args.save_dir, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
    fdir = "%s/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_fold%d/" % (
        args.save_dir,
        args.mlr,
        args.lr,
        args.lambda_l2,
        args.model_type,
        args.num_epoch,
        args.batch_size_vl,
        args.opt_type,
        args.update_freq,
        args.reset_freq,
        args.rng,
    )

    os.makedirs(fdir, exist_ok=True)
    os.makedirs(fdir + "/checkpoint/", exist_ok=True)
    print(fdir)
    train(args, dataset, model, optimizer, fdir)


def train(args: Config, dataset, model, optimizer, fdir):
    counter = 0
    tr_loss_list, tr_acc_list, vl_loss_list = [], [], []
    tr_corr_mean_list, tr_corr_std_list = [], []
    optimizer = update_optimizer_hyperparams(model, optimizer)

    for epoch in range(args.num_epoch + 1):
        if epoch % args.test_freq == 0:
            with torch.no_grad():
                ds = (to_torch_variable(data, target, args.is_cuda, floatTensorF=1) for data, target in dataset[TEST])
                results = [evaluate(data, target, model) for data, target in ds]
                te_loss = np.mean([loss.item() for loss, _ in results])
                te_acc = np.mean([accuracy for _, accuracy in results])

                wandb.log(
                    {
                        "test_epoch": epoch,
                        "test_loss": te_loss,
                        "test_accuracy": te_acc,
                    }
                )
                print("Valid Epoch: %d, Loss %f Acc %f" % (epoch, te_loss, te_acc))

        grad_list = []
        start_time = time.time()
        for data, target in dataset[TRAIN]:
            data_vl, target_vl = next(dataset[VALID])
            data, target = to_torch_variable(data, target, args.is_cuda)
            data_vl, target_vl = to_torch_variable(data_vl, target_vl, args.is_cuda)

            unupdated = deepcopy(model)
            optimizer.zero_grad()
            model, loss, accuracy = feval(data, target, model)
            optimizer.step()

            model, optimizer, loss_vl, acc_vl = meta_update(
                args, data_vl, target_vl, data, target, model, optimizer, unupdated
            )

            wandb.log(
                {
                    "train_epoch": counter,
                    "train_loss": loss,
                    "train_accuracy": accuracy,
                    "valid_loss": loss_vl,
                    "valid_epoch": counter,
                    "learning_rate": F.softplus(torch.tensor(model.eta)).item(),
                    "weight_decay": F.softplus(torch.tensor(model.lambda_l2)).item(),
                    "dFdlr_norm": model.dFdlr_norm,
                    "dFdl2_norm": model.dFdl2_norm,
                    "grad_norm": model.grad_norm,
                    "grad_norm_vl": model.grad_norm_vl,
                    "grad_angle": model.grad_angle,
                    "param_norm": model.param_norm,
                    "valid_accuracy": acc_vl,
                }
            )

            tr_loss_list.append(loss)
            tr_acc_list.append(accuracy)
            vl_loss_list.append(loss_vl)
            grad_vec = flatten([p.grad.data for p in model.parameters()]).data.cpu().numpy()
            grad_list.append(grad_vec / np.linalg.norm(grad_vec))
            counter += 1

        corr_mean, corr_std = compute_correlation(grad_list, normF=1)
        tr_corr_mean_list.append(corr_mean)
        tr_corr_std_list.append(corr_std)

        end_time = time.time()
        if epoch == 0:
            print("Single epoch timing %f" % ((end_time - start_time) / 60))

        if epoch % args.checkpoint_freq == 0:
            os.makedirs(fdir + "/checkpoint/", exist_ok=True)
            save_object_as_wandb_artifact(
                [model.state_dict()], f"model_{wandb.run.id}", f"{fdir}/checkpoint", f"epoch{epoch}", "model"
            )

        wandb.log(
            {
                "train_epoch": counter,
                "corr_mean": corr_mean,
                "corr_std": corr_std,
            }
        )

        fprint = "Train Epoch: %d, Tr Loss %f Vl loss %f Acc %f Eta %s, L2 %s, |dFdlr| %.2f |dFdl2| %.2f |G| %.4f |G_vl| %.4f Gang %.3f |W| %.2f, Grad Corr %f %f"
        print(
            fprint
            % (
                epoch,
                np.mean(tr_loss_list[-100:]),
                np.mean(vl_loss_list[-100:]),
                np.mean(tr_acc_list[-100:]),
                str(F.softplus(torch.tensor(model.eta)).item()),
                str(F.softplus(torch.tensor(model.lambda_l2)).item()),
                model.dFdlr_norm,
                model.dFdl2_norm,
                model.grad_norm,
                model.grad_norm_vl,
                model.grad_angle,
                model.param_norm,
                corr_mean,
                corr_std,
            )
        )


def evaluate(data, target, model):
    output = model(data)
    loss = F.nll_loss(output, target)
    pred = output.argmax(dim=1, keepdim=True).flatten()  # get the index of the max log-probability
    accuracy = pred.eq(target).float().mean()

    return loss, accuracy.item()


def flatten(xs):
    return torch.cat([x.detach().flatten() for x in xs])


def feval(data, target, model):
    model.train()
    loss, accuracy = evaluate(data, target, model)
    loss.backward()

    grad = flatten([p.grad.data for p in model.parameters()])
    param = flatten(model.parameters())
    model.grad_norm = torch.linalg.norm(grad).item()
    model.param_norm = torch.linalg.norm(param).item()

    return model, loss.item(), accuracy


def compute_HessianVectorProd(args: Config, dFdS, data, target, unupdated):
    Hess_est_r = args.hv_r

    def perturb(model, vector):
        current_params = parameters_to_vector(model.parameters())
        new_params = current_params + vector.to(current_params.device)
        vector_to_parameters(new_params, model.parameters())
        return feval(data, target, model)

    model_plus = deepcopy(unupdated)
    model_minus = deepcopy(unupdated)

    model_plus, _, _ = perturb(model_plus, Hess_est_r * dFdS)
    model_minus, _, _ = perturb(model_minus, -Hess_est_r * dFdS)

    g_plus = flatten([p.grad.data for p in model_plus.parameters()])
    g_minus = flatten([p.grad.data for p in model_minus.parameters()])
    Hv = (g_plus - g_minus) / (2 * Hess_est_r)
    return Hv


def meta_update(args: Config, data_vl, target_vl, data_tr, target_tr, model, optimizer, unupdated):
    # Compute Hessian Vector Product
    Hv_lr = compute_HessianVectorProd(args, model.dFdlr, data_tr, target_tr, unupdated)
    Hv_l2 = compute_HessianVectorProd(args, model.dFdl2, data_tr, target_tr, unupdated)

    val_model = deepcopy(model)
    val_model.train()
    _, loss_valid, acc_valid = feval(data_vl, target_vl, val_model)
    grad_valid = flatten([p.grad.data for p in val_model.parameters()])
    model.grad_norm_vl = torch.linalg.norm(grad_valid).item()

    # Compute angle between tr and vl grad

    grad = flatten([p.grad.data for p in model.parameters()])
    param = flatten(model.parameters())
    model.grad_angle = torch.dot(grad / model.grad_norm, grad_valid / model.grad_norm_vl).item()

    # Update hyper-parameters
    model.update_dFdlr(Hv_lr, param, grad)
    model.update_dFdlambda_l2(Hv_l2, param)

    # model.update_eta(args.mlr, grad_valid)
    # model.update_lambda(args.mlr, grad_valid)

    delta_eta = grad_valid.dot(model.dFdlr).item()
    delta_lambda = grad_valid.dot(model.dFdl2).item()
    model.update_hyperparams(delta_eta, delta_lambda, mlr=args.mlr, method=args.meta_optimizer)

    # model.update_hyperparams(delta_eta, delta_lambda, mlr=1e-3, method='adam')

    # Update optimizer with new eta
    optimizer = update_optimizer_hyperparams(model, optimizer)

    return model, optimizer, loss_valid, acc_valid


def update_optimizer_hyperparams(model, optimizer):
    optimizer.param_groups[0]["lr"] = np.copy(F.softplus(torch.tensor(model.eta)))
    optimizer.param_groups[0]["weight_decay"] = F.softplus(torch.tensor(model.lambda_l2))

    return optimizer


if __name__ == "__main__":
    sweep_config = get_sweep_config()
    if not sweep_config:
        raise ValueError("No sweep config received")

    wandb_kwargs = {
        "mode": "offline",
        "group": sweep_config.name,
        "config": sweep_config.config,
        "project": sweep_config.config["project"],
    }

    with wandb.init(**wandb_kwargs) as run:
        args = Config(**run.config)
        if args.use_64:
            torch.set_default_dtype(torch.float64)
        seed = args.rng
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        main(args)
