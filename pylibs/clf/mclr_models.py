#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: Santosh Kesiraju
# @email : kcraj2@gmail.com, kesiraju@fit.vutbr.cz

"""
Multi-class Logistic Regression exploiting uncertainty in the input features.
"""


import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


class MCLR(nn.Module):
    """Multi-class Logistic Regression"""

    def __init__(
        self, dim, n_classes, lam_w=5e-02, trn_iters=500, lr=1e-02, cuda: bool = False
    ):
        """initialize the model

        Args:
        ----
            dim (int): Input feature dimension
            n_classes (int): Number of classes
            lam_w (float): L2 regularization weight for the parameters
            trn_iters (int): Training iterations
            lr (float): Learning rate
            cuda (bool): CUDA ?
        """

        super(MCLR, self).__init__()

        self.device = torch.device("cuda" if cuda else "cpu")

        self.b = nn.Parameter(torch.randn(1, n_classes) * 0.001, requires_grad=True)
        self.W = nn.Parameter(torch.randn(dim, n_classes) * 0.001, requires_grad=True)
        self.lam_w = nn.Parameter(torch.Tensor([lam_w]), requires_grad=False)

        self.dim = nn.Parameter(torch.LongTensor([dim]), requires_grad=False)

        self.trn_iters = trn_iters
        self.lr = lr

        self.xen_loss = nn.CrossEntropyLoss(reduction="sum")
        self.log_softmax = nn.LogSoftmax(dim=1)

    def compute_grads(self, compute: bool):
        """compute grads, yes or no"""

        self.W.requires_grad_(compute)
        self.b.requires_grad_(compute)

    def forward(self, X):
        """forward

        Args:
        ----
            X (torch.Tensor): [n_samples, feat_dim]

        Returns:
        --------
            torch.Tensor: logits
        """

        logits = (X @ self.W) + self.b
        return logits

    # def reg_weights(self):
    #    """ L2 regularization for weights """
    #    return (self.W ** 2).sum() * self.lam_w

    def loss(self, logits, Y):
        """Compute loss"""

        xen = self.xen_loss(logits, Y)
        return xen

    def fit(self, X, Y):
        """Fit or train the model"""

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y).to(dtype=torch.long)

        # if opt  == 'adam':
        # print("MCLR: Using Adam with default learning rate.")
        opt = torch.optim.Adam([self.W, self.b], lr=self.lr, weight_decay=self.lam_w)

        train(self, opt, X, Y, self.trn_iters)

    def fit_and_validate(
        self, x_train, y_train, x_dev, y_dev, out_sfx="", val_iters=1, save=True
    ):

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y).to(dtype=torch.long)

        # if opt == 'adam':
        # print("MCLR: Using adam with default learning rate.")
        opt = torch.optim.Adam([self.W, self.b], lr=self.lr, weight_decay=self.lam_w)

        train_and_validate(
            self,
            x_train,
            y_train,
            x_dev,
            y_dev,
            opt,
            out_sfx,
            self.trn_iters,
            val_iters,
            save,
        )

    def predict_proba(self, X):
        """Predict posterior probability of class labels"""

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(device=self.device)

        self.compute_grads(False)
        logits = self.forward(X)
        return torch.exp(self.log_softmax(logits))

    def predict(self, X):

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(device=self.device)

        return predict(self, X)


class MCLRU(nn.Module):
    """Multi-class Logistic Regression with uncertainty"""

    def __init__(
        self,
        dim,
        n_classes,
        lam_w=5e-02,
        trn_iters=500,
        lr=1e-02,
        R=128,
        cuda=False,
        use_uncert=True,
    ):
        """initialize the model

        Args:
        ----
            dim (int): Input feature dimension
            n_classes (int): Number of classes
            lam_w (float): L2 regularization weight for the parameters
            trn_iters (int): Training iterations
            lr (float): Learning rate
            R (int): number of Monte Carlo samples
            cuda (bool): CUDA ?
            use_uncert (bool): Use uncertainties during training ?
        """

        super(MCLRU, self).__init__()

        self.device = torch.device("cuda" if cuda else "cpu")
        self.use_uncert = use_uncert

        self.b = nn.Parameter(torch.randn(1, n_classes), requires_grad=True)
        self.W = nn.Parameter(torch.randn(dim, n_classes), requires_grad=True)
        nn.init.xavier_uniform_(self.b.data)
        nn.init.xavier_uniform_(self.W.data)
        self.lam_w = nn.Parameter(torch.Tensor([lam_w]), requires_grad=False)

        self.R = nn.Parameter(torch.Tensor([R]), requires_grad=False).long()
        self.dim = nn.Parameter(torch.LongTensor([dim]), requires_grad=False)

        self.trn_iters = trn_iters
        self.lr = lr

        self.xen_loss = nn.CrossEntropyLoss(reduction="sum")
        self.log_softmax = nn.LogSoftmax(dim=2)

    def compute_grads(self, compute: bool):
        """compute grads, yes or no"""

        self.W.requires_grad_(compute)
        self.b.requires_grad_(compute)

    def sample(self, N):
        """Generate samples from std. Normal"""

        return torch.randn(size=(self.R, N, self.dim)).to(device=self.device)

    def forward(self, X: torch.Tensor):
        """forward

        Args:
        ----
            X (torch.Tensor): [n_samples, mean;log_std]
        """

        if self.use_uncert:
            eps = self.sample(X.shape[0])
            # transformation: mean + (eps * std)
            X_1 = X[:, : self.dim] + (eps * torch.exp(X[:, self.dim :]))
        else:
            X_1 = X[:, : self.dim]

        logits = (X_1 @ self.W) + self.b
        return logits

    # def reg_weights(self):
    #    """ L2 regularization for weights """
    #    return (self.W ** 2).sum() * self.lam_w

    def loss(self, logits, Y):
        """Compute loss"""

        xen = torch.Tensor([0.0]).to(device=self.device)
        if self.use_uncert:
            for i in range(self.R):
                xen += self.xen_loss(logits[i, :, :], Y)
            xen /= self.R.float().to(device=self.device)
        else:
            xen += self.xen_loss(logits, Y)

        return xen

    def fit(self, X, Y):
        """Fit or train the model"""

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y).to(dtype=torch.long)

        # if opt == 'adam':
        # print("MCLRU: Using Adam with default learning rate.")
        opt = torch.optim.Adam([self.W, self.b], lr=self.lr, weight_decay=self.lam_w)

        train(self, opt, X, Y, self.trn_iters)

    def fit_and_validate(
        self, x_train, y_train, x_dev, y_dev, out_sfx="", val_iters=1, save=True
    ):

        if isinstance(x_train, np.ndarray):
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train).to(dtype=torch.long)
            x_dev = torch.from_numpy(x_dev)
            y_dev = torch.from_numpy(y_dev).to(dtype=torch.long)

        # if opt == 'adam':
        # print("MCRLU: Using Adam with default learning rate.")
        opt = torch.optim.Adam([self.W, self.b], lr=self.lr, weight_decay=self.lam_w)

        train_and_validate(
            self,
            x_train,
            y_train,
            x_dev,
            y_dev,
            opt,
            out_sfx,
            self.trn_iters,
            val_iters,
            save,
        )

    def predict_proba(self, X):
        """Predict posterior probability of class labels"""

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(device=self.device)

        self.compute_grads(False)
        self.use_uncert = True

        logits = self.forward(X)
        return torch.exp(self.log_softmax(logits)).mean(dim=0)

    def predict(self, X):

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(device=self.device)

        return predict(self, X)


def save_model(model, out_file):
    """Save model"""

    logger.info("Saving model %s", out_file)
    torch.save(model.state_dict(), out_file)


def load_model(model, model_file):
    """Load model"""

    model.load_state_dict(torch.load(model_file))
    return model


def train(model, optim, X, Y, trn_iters):
    """Train the model"""

    train_dset = torch.utils.data.TensorDataset(X, Y)
    train_dataloader = torch.utils.data.DataLoader(
        train_dset, batch_size=4096, pin_memory=True
    )

    # train_losses = []
    best_loss = torch.Tensor([9999999]).to(model.device)

    for i in range(trn_iters):

        total_loss = torch.tensor(0.0).to(device=model.device)
        for x, y in train_dataloader:
            x = x.to(device=model.device)
            y = y.to(device=model.device)

            optim.zero_grad()
            logits = model(x)
            xen = model.loss(logits, y)
            xen.backward()
            optim.step()
            total_loss += xen.detach().item()

        # train_losses.append(xen.detach().item())
        info_str = "Iter {:4d} / {:4d} Loss: {:f}".format(
            i + 1, trn_iters, total_loss.detach().cpu().item()
        )
        logger.info(info_str)

        if best_loss > total_loss.detach().item():
            best_loss = total_loss.detach().item()

        else:
            logger.info("Early stopping..")
            break

    model.compute_grads(False)
    return model


def train_and_validate(
    model,
    x_train,
    y_train,
    x_dev,
    y_dev,
    optim,
    out_sfx,
    trn_iters,
    val_iters=1,
    save=True,
):
    """Train the model and validate after `val_iters`"""

    x_train = x_train.to(device=model.device)
    y_train = y_train.to(device=model.device)

    x_dev = x_dev.to(device=model.device)
    y_dev = y_dev.to(device=model.device)

    # first col: train_acc, second col: dev_acc
    scores = np.zeros(shape=(trn_iters, 2), dtype=np.float32)

    best_dev_acc = 0.001
    best_model_file = ""

    for i in range(trn_iters):

        optim.zero_grad()
        logits = model.forward(x_train)
        xen = model.loss(logits, y_train)
        xen.backward()
        optim.step()

        if (i + 1) % val_iters == 0:

            # turn of gradient computations
            model.compute_grads(False)

            scores[i, 0] = (
                np.mean(predict(model, x_train) == y_train.cpu().numpy()) * 100
            )
            scores[i, 1] = np.mean(predict(model, x_dev) == y_dev.cpu().numpy()) * 100

            if scores[i, 1] > best_dev_acc:
                best_dev_acc = scores[i, 1]
                if save:
                    best_model_file = out_sfx + f"_{i+1}.pt"
                    # save_model(model, best_model_file)
                    torch.save(model.state_dict(), best_model_file)

            # turn on gradient computations
            model.compute_grads(True)

            logger.info(
                "Iter {:4d}/{:4d} Loss: {:.2f} "
                "Train acc: {:.2f} Dev acc: {:.2f}".format(
                    i + 1, trn_iters, xen.detach().cpu().numpy().item(), *scores[i, :]
                )
            )

    return best_model_file, scores


def predict(model, X):
    """Predict post. prob of classes given the features"""

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    X = X.to(device=model.device)

    probs = model.predict_proba(X)
    return torch.argmax(probs, dim=1).cpu().numpy()


def main():
    """main method"""

    args = parse_arguments()

    clf = MCLRU(5, 4, R=args.R, cuda=False)

    X = torch.randn(100, 5 * 2)
    Y = torch.randint(0, 4, size=(100,))

    X_dev = torch.randn(100, 5 * 2)
    Y_dev = torch.randint(0, 4, size=(100,))

    opt = torch.optim.Adam(clf.parameters())

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        filename=os.path.join("/tmp/", f"run.log"),
        filemode="w",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # train(clf, opt, X, Y, 5)

    # clf.fit(X, Y, opt, 5)

    clf.fit_and_validate(X, Y, X_dev, Y_dev, opt, trn_iters=5)


def parse_arguments():
    """parse command line args"""

    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("train_feats_f", help="path to training feats file")
    # parser.add_argument("train_labels_f", help="path to training labels file")
    # parser.add_argument("out_dir", help="out dir to save the classifier, results")
    parser.add_argument("-trn", default=10, type=int, help="training iters")
    parser.add_argument(
        "-lw", default=1e-4, type=float, help="L2 reg weight for the model params"
    )
    parser.add_argument(
        "-R", default=1, type=int, help="number of samples for Monte Carlo approx"
    )
    parser.add_argument("--nocuda", action="store_true", help="do not use cuda")

    args = parser.parse_args()

    args.cuda = bool(torch.cuda.is_available() and not args.nocuda)

    return args


if __name__ == "__main__":
    main()
