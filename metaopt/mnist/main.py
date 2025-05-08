from dataclasses import dataclass
import os, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
import hashlib
from metaopt.mnist.mlp import MLP
from metaopt.util import check_args, to_torch_variable, save
from metaopt.util_ml import compute_correlation
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sweep_agent.agent import get_sweep_config
import wandb
import joblib


@dataclass
class Config:
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
    save_dir: str = "/scratch/"


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
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1) if logsoftmaxF else F.softmax(out, dim=1)

    def update_dFdlr(self, Hv, param, grad):
        self.Hlr = self.eta * Hv
        self.Hlr_norm = torch.norm(self.Hlr)
        self.dFdlr_norm = torch.norm(self.dFdlr)
        self.dFdlr.data = (
            self.dFdlr.data * (1 - 2 * self.lambda_l2 * self.eta) - self.Hlr - grad - 2 * self.lambda_l2 * param
        )

    def update_dFdlambda_l2(self, Hv, param):
        self.Hl2 = self.eta * Hv
        self.Hl2_norm = torch.norm(self.Hl2)
        self.dFdl2_norm = torch.norm(self.dFdl2)
        self.dFdl2.data = self.dFdl2.data * (1 - 2 * self.lambda_l2 * self.eta) - self.Hl2 - 2 * self.eta * param

    def update_eta(self, mlr, val_grad):
        delta = val_grad.dot(self.dFdlr).data.cpu().numpy()
        self.eta -= mlr * delta
        self.eta = max(0.0, self.eta)

    def update_lambda(self, mlr, val_grad):
        val_grad = torch.cat([v.view(-1) for v in val_grad])
        delta = val_grad.dot(self.dFdl2).data.cpu().numpy()
        self.lambda_l2 -= mlr * delta
        self.lambda_l2 = np.clip(self.lambda_l2, 0, 0.0002)


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
        self.Hlr = self.eta * Hv
        self.Hlr_norm = torch.norm(self.Hlr)
        self.dFdlr_norm = torch.norm(self.dFdlr)
        self.dFdlr.data = (
            self.dFdlr.data * (1 - 2 * self.lambda_l2 * self.eta) - self.Hlr - grad - 2 * self.lambda_l2 * param
        )

    def update_dFdlambda_l2(self, Hv, param):
        self.Hl2 = self.eta * Hv
        self.Hl2_norm = torch.norm(self.Hl2)
        self.dFdl2_norm = torch.norm(self.dFdl2)
        self.dFdl2.data = self.dFdl2.data * (1 - 2 * self.lambda_l2 * self.eta) - self.Hl2 - 2 * self.eta * param

    def update_eta(self, mlr, val_grad):
        delta = val_grad.dot(self.dFdlr).data.cpu().numpy()
        self.eta -= mlr * delta
        self.eta = max(0.0, self.eta)

    def update_lambda(self, mlr, val_grad):
        val_grad = torch.cat([v.view(-1) for v in val_grad])
        delta = val_grad.dot(self.dFdl2).data.cpu().numpy()
        self.lambda_l2 -= mlr * delta
        self.lambda_l2 = np.clip(self.lambda_l2, 0, 0.0002)


TRAIN = 0
VALID = 1
TEST = 2


def compute_md5(file_path):
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compare_npy_files(new_dir, ref_dir, file_names):
    """Compare .npy files in new_dir against reference files in ref_dir."""
    mismatched_files = []
    for file_name in file_names:
        new_file = os.path.join(new_dir, file_name + ".npy")
        ref_file = os.path.join(ref_dir, file_name + ".npy")

        if not os.path.exists(ref_file):
            print(f"Reference file {ref_file} does not exist.")
            continue
        if not os.path.exists(new_file):
            print(f"New file {new_file} does not exist.")
            continue

        new_hash = compute_md5(new_file)
        ref_hash = compute_md5(ref_file)

        if new_hash != ref_hash:
            print(f"Hash mismatch in {file_name}: New hash {new_hash}, Ref hash {ref_hash}")

            # Load and compare the .npy files
            try:
                new_data = np.load(new_file)
                ref_data = np.load(ref_file)

                if new_data.shape != ref_data.shape:
                    print(f"  Shape mismatch: New shape {new_data.shape}, Ref shape {ref_data.shape}")
                    mismatched_files.append((file_name, new_hash, ref_hash))
                elif not np.allclose(new_data, ref_data, rtol=1e-5, atol=1e-8):
                    # Find indices and values where arrays differ significantly
                    diff_indices = np.where(~np.isclose(new_data, ref_data, rtol=1e-5, atol=1e-8))
                    print(f"  Arrays differ at {len(diff_indices[0])} positions:")
                    for idx in zip(*diff_indices):
                        print(f"    Index {idx}: New value {new_data[idx]}, Ref value {ref_data[idx]}")
                    mismatched_files.append((file_name, new_hash, ref_hash))
                else:
                    print(
                        "  Arrays are equivalent within tolerance despite hash mismatch (possible metadata difference)."
                    )
            except Exception as e:
                print(f"  Error loading .npy files for comparison: {e}")
                mismatched_files.append((file_name, new_hash, ref_hash))
        else:
            print(f"Match confirmed for {file_name}: Hash {new_hash}")

    if mismatched_files:
        print("Warning: The following files do not match the reference:")
        for file_name, new_hash, ref_hash in mismatched_files:
            print(f"{file_name}: New hash {new_hash}, Ref hash {ref_hash}")
        raise ValueError("Refactored code produces different .npy files.")
    else:
        print("All .npy files match the reference files or are within tolerance.")


def save_object_as_wandb_artifact(obj, artifact_name: str, fdir: str, filename: str, artifact_type: str) -> None:
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"

    full_path = os.path.join(fdir, filename)
    joblib.dump(obj, full_path, compress=0)
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(full_path)
    wandb.log_artifact(artifact)


def load_mnist(args: Config):
    ## Initialize Dataset
    dataset = datasets.MNIST(
        "data/mnist", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )
    train_set, valid_set = torch.utils.data.random_split(dataset, [60000 - args.valid_size, args.valid_size])

    data_loader_tr = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    data_loader_vl = DataLoader(valid_set, batch_size=args.batch_size_vl, shuffle=True)
    data_loader_te = DataLoader(
        datasets.MNIST("data/mnist", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size,
        shuffle=True,
    )

    data_loader_vl = cycle(data_loader_vl)
    dataset = [data_loader_tr, data_loader_vl, data_loader_te]
    return dataset


def main(args: Config):
    dataset = load_mnist(args)

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

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)

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
                    "learning_rate": model.eta,
                    "weight_decay": model.lambda_l2,
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
                str(model.eta),
                str(model.lambda_l2),
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

    return model, loss.item(), accuracy


def compute_HessianVectorProd(dFdS, data, target, unupdated):
    Hess_est_r = 1e-3

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


def meta_update(args, data_vl, target_vl, data_tr, target_tr, model, optimizer, unupdated):
    # Compute Hessian Vector Product
    Hv_lr = compute_HessianVectorProd(model.dFdlr, data_tr, target_tr, unupdated)
    # Hv_l2 = compute_HessianVectorProd(model.dFdl2, data_tr, target_tr, unupdated, is_cuda)

    val_model = deepcopy(model)
    val_model.train()
    _, loss_valid, acc_valid = feval(data_vl, target_vl, val_model)
    grad_valid = flatten([p.grad.data for p in val_model.parameters()])
    model.grad_norm_vl = torch.linalg.norm(grad_valid).item()

    # Compute angle between tr and vl grad
    grad = flatten([p.grad.data for p in model.parameters()])
    param = flatten(model.parameters())
    model.grad_norm = torch.linalg.norm(grad).item()
    model.param_norm = torch.linalg.norm(param).item()
    model.grad_angle = torch.dot(grad / model.grad_norm, grad_valid / model.grad_norm_vl).item()

    # Update hyper-parameters
    model.update_dFdlr(Hv_lr, param, grad)
    model.update_eta(args.mlr, grad_valid)
    # Update optimizer with new eta
    optimizer = update_optimizer_hyperparams(model, optimizer)

    return model, optimizer, loss_valid, acc_valid


def update_optimizer_hyperparams(model, optimizer):
    optimizer.param_groups[0]["lr"] = np.copy(model.eta)
    optimizer.param_groups[0]["weight_decay"] = model.lambda_l2

    return optimizer


if __name__ == "__main__":
    sweep_config = get_sweep_config()
    if not sweep_config:
        raise ValueError("No sweep config received")

    wandb_kwargs = {
        "mode": "offline",
        "group": sweep_config.name,
        "config": sweep_config.config,
        "project": "new_metaopt",
    }

    with wandb.init(**wandb_kwargs) as run:
        args = Config(**run.config)
        torch.manual_seed(args.rng)
        main(args)
