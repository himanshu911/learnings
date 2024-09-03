# %%
__all__ = [
    "Dataset",
    "get_dls",
    "collate_dict",
    "DataLoaders",
    "show_image",
    "subplots",
    "get_grid",
    "show_images",
    "to_device",
    "to_cpu",
    "collate_device",
    "CancelFitException",
    "CancelBatchException",
    "CancelEpochException",
    "Callback",
    "run_cbs",
    "with_cbs",
    "Learner",
    "TrainLearner",
    "TrainCB",
    "DeviceCB",
    "SingleBatchCB",
    "MetricsCB",
    "ProgressCB",
    "LRFinderCB",
    "lr_find",
    "BaseSchedCB",
    "BatchSchedCB",
    "HasLearnCB",
    "RecorderCB",
    "EpochSchedCB",
    "MixedPrecision",
    "AccelerateCB",
]

# %%
import math, typing, time
from collections.abc import Mapping
from copy import copy
from itertools import zip_longest
from functools import partial, wraps
from operator import attrgetter, itemgetter

import matplotlib.pyplot as plt
import numpy as np
import fastcore.all as fc
from fastprogress import progress_bar, master_bar
from fastprogress.fastprogress import format_time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import default_collate

from torcheval.metrics import Mean
from accelerate import Accelerator


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, **kwargs),
    )


class Dataset:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def collate_dict(ds):
    get = itemgetter(*ds.features)

    def _f(b):
        return get(default_collate(b))

    return _f


class DataLoaders:
    def __init__(self, *dls):
        self.train, self.valid = dls[:2]

    @classmethod
    def from_dd(cls, dd, batch_size, as_tuple=True, **kwargs):
        f = collate_dict(dd["train"])
        return cls(*get_dls(*dd.values(), bs=batch_size, collate_fn=f))


@fc.delegates(plt.Axes.imshow)
def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    if fc.hasattrs(im, ("cpu", "permute", "detach")):
        im = im.detach().cpu()
        if len(im.shape) == 3 and im.shape[0] < 5:
            im = im.permute(1, 2, 0)
    elif not isinstance(im, np.ndarray):
        im = np.array(im)
    if im.shape[-1] == 1:
        im = im[..., 0]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if noframe:
        ax.axis("off")
    return ax


@fc.delegates(plt.subplots, keep=True)
def subplots(
    nrows: int = 1,  # Number of rows in returned axes grid
    ncols: int = 1,  # Number of columns in returned axes grid
    figsize: tuple = None,  # Width, height in inches of the returned figure
    imsize: int = 3,  # Size (in inches) of images that will be displayed in the returned figure
    suptitle: str = None,  # Title to be set to returned figure
    **kwargs,
):  # fig and axs
    "A figure and set of subplots to display images of `imsize` inches"
    if figsize is None:
        figsize = (ncols * imsize, nrows * imsize)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None:
        fig.suptitle(suptitle)
    if nrows * ncols == 1:
        ax = np.array([ax])
    return fig, ax


@fc.delegates(subplots)
def get_grid(
    n: int,  # Number of axes
    nrows: int = None,  # Number of rows, defaulting to `int(math.sqrt(n))`
    ncols: int = None,  # Number of columns, defaulting to `ceil(n/rows)`
    title: str = None,  # If passed, title set to the figure
    weight: str = "bold",  # Title font weight
    size: int = 14,  # Title font size
    **kwargs,
):  # fig and axs
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows:
        ncols = ncols or int(np.floor(n / nrows))
    elif ncols:
        nrows = nrows or int(np.ceil(n / ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n / nrows))
    fig, axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows * ncols):
        axs.flat[i].set_axis_off()
    if title is not None:
        fig.suptitle(title, weight=weight, size=size)
    return fig, axs


@fc.delegates(subplots)
def show_images(
    ims: list,  # Images to show
    nrows: typing.Union[int, None] = None,  # Number of rows in grid
    ncols: typing.Union[
        int, None
    ] = None,  # Number of columns in grid (auto-calculated if None)
    titles: typing.Union[list, None] = None,  # Optional list of titles for each image
    **kwargs,
):
    "Show all images `ims` as subplots with `rows` using `titles`"
    axs = get_grid(len(ims), nrows, ncols, **kwargs)[1].flat
    for im, t, ax in zip_longest(ims, titles or [], axs):
        show_image(im, ax=ax, title=t)


def_device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def to_device_general(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: to_device_general(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device_general(o, device) for o in x)
    return x


# Define an MPS-specific to_device function
def to_device_mps(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)  # Converts tensor to float32 for MPS compatibility
    if isinstance(x, Mapping):
        return {k: to_device_mps(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device_mps(o, device) for o in x)
    return x


# Check if MPS is available and decide which to_device function to use
if torch.backends.mps.is_available():
    to_device = to_device_mps
else:
    to_device = to_device_general


def to_cpu(x):
    if isinstance(x, Mapping):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cpu(o) for o in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(list(x)))
    return x.detach().cpu()


def collate_device(b):
    return to_device(default_collate(b))


class CancelFitException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class Callback:
    order = 0


def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter("order")):
        method = getattr(cb, method_nm, None)
        if method is not None:
            method(learn)


class with_cbs:
    def __init__(self, nm):
        self.nm = nm

    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f"before_{self.nm}")
                f(o, *args, **kwargs)
                o.callback(f"after_{self.nm}")
            except globals()[f"Cancel{self.nm.title()}Exception"]:
                pass
            finally:
                o.callback(f"cleanup_{self.nm}")

        return _f


class Learner:
    def __init__(
        self,
        model,
        dls=(0,),
        loss_func=F.mse_loss,
        lr=0.1,
        cbs=None,
        opt_func=optim.SGD,
    ):
        cbs = fc.L(cbs)
        fc.store_attr()

    @with_cbs("batch")
    def _one_batch(self):
        self.predict()
        self.callback("after_predict")
        self.get_loss()
        self.callback("after_loss")
        if self.training:
            self.backward()
            self.callback("after_backward")
            self.step()
            self.callback("after_step")
            self.zero_grad()

    @with_cbs("epoch")
    def _one_epoch(self):
        for self.iter, self.batch in enumerate(self.dl):
            self._one_batch()

    def one_epoch(self, training):
        self.model.train(training)
        self.dl = self.dls.train if training else self.dls.valid
        self._one_epoch()

    @with_cbs("fit")
    def _fit(self, train, valid):
        for self.epoch in self.epochs:
            if train:
                self.one_epoch(True)
            if valid:
                torch.no_grad()(self.one_epoch)(False)

    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None):
        cbs = fc.L(cbs)
        # `add_cb` and `rm_cb` were added in lesson 18
        for cb in cbs:
            self.cbs.append(cb)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            if lr is None:
                lr = self.lr
            if self.opt_func:
                self.opt = self.opt_func(self.model.parameters(), lr)
            self._fit(train, valid)
        finally:
            for cb in cbs:
                self.cbs.remove(cb)

    def __getattr__(self, name):
        if name in ("predict", "get_loss", "backward", "step", "zero_grad"):
            return partial(self.callback, name)
        raise AttributeError(name)

    def callback(self, method_nm):
        run_cbs(self.cbs, method_nm, self)

    @property
    def training(self):
        return self.model.training


class TrainLearner(Learner):
    def predict(self):
        self.preds = self.model(self.batch[0])

    def get_loss(self):
        self.loss = self.loss_func(self.preds, self.batch[1])

    def backward(self):
        self.loss.backward()

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


class TrainCB(Callback):
    def __init__(self, n_inp=1):
        self.n_inp = n_inp

    def predict(self, learn):
        learn.preds = learn.model(*learn.batch[: self.n_inp])

    def get_loss(self, learn):
        learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inp :])

    def backward(self, learn):
        learn.loss.backward()

    def step(self, learn):
        learn.opt.step()

    def zero_grad(self, learn):
        learn.opt.zero_grad()


class DeviceCB(Callback):
    def __init__(self, device=def_device):
        fc.store_attr()

    def before_fit(self, learn):
        if hasattr(learn.model, "to"):
            learn.model.to(self.device)

    def before_batch(self, learn):
        learn.batch = to_device(learn.batch, device=self.device)


class SingleBatchCB(Callback):
    order = 1

    def after_batch(self, learn):
        raise CancelFitException()


class MetricsCB(Callback):
    def __init__(self, *ms, **metrics):
        for o in ms:
            metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics["loss"] = self.loss = Mean()
        self.res = []

    def _log(self, d):
        print(d)

    def before_fit(self, learn):
        learn.metrics = self

    def before_epoch(self, learn):
        self.start_epoch = time.time()
        [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, learn):
        log = {k: f"{v.compute():.3f}" for k, v in self.all_metrics.items()}
        log["epoch"] = learn.epoch
        log["train"] = "train" if learn.model.training else "eval"
        log["time"] = format_time(time.time() - self.start_epoch)
        self.res.append(log)
        self._log(log)

    def after_batch(self, learn):
        x, y, *_ = to_cpu(learn.batch)
        for m in self.metrics.values():
            m.update(to_cpu(learn.preds), y)
        self.loss.update(to_cpu(learn.loss), weight=len(x))


class ProgressCB(Callback):
    order = MetricsCB.order + 1

    def __init__(self, plot=False):
        self.plot = plot

    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, "metrics"):
            learn.metrics._log = self._log
        self.losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn):
        learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)

    def after_batch(self, learn):
        learn.dl.comment = f"{learn.loss:.3f}"
        if self.plot and hasattr(learn, "metrics") and learn.training:
            self.losses.append(learn.loss.item())
            self.mbar.update_graph([[fc.L.range(self.losses), self.losses]])


class LRFinderCB(Callback):
    def __init__(self, gamma=1.3, max_mult=3):
        fc.store_attr()

    def before_fit(self, learn):
        self.sched = ExponentialLR(learn.opt, self.gamma)
        self.lrs, self.losses = [], []
        self.min = math.inf

    def after_batch(self, learn):
        if not learn.training:
            raise CancelEpochException()
        self.lrs.append(learn.opt.param_groups[0]["lr"])
        loss = to_cpu(learn.loss)
        self.losses.append(loss)
        if loss < self.min:
            self.min = loss
        if loss > self.min * self.max_mult:
            raise CancelFitException()
        self.sched.step()

    def cleanup_fit(self, learn):
        plt.plot(self.lrs, self.losses)
        plt.xscale("log")


@fc.patch
def lr_find(self: Learner, gamma=1.3, max_mult=3, start_lr=1e-5, max_epochs=10):
    self.fit(max_epochs, lr=start_lr, cbs=LRFinderCB(gamma=gamma, max_mult=max_mult))


class BaseSchedCB(Callback):
    def __init__(self, sched):
        self.sched = sched

    def before_fit(self, learn):
        self.schedo = self.sched(learn.opt)

    def _step(self, learn):
        if learn.training:
            self.schedo.step()


class BatchSchedCB(BaseSchedCB):
    def after_batch(self, learn):
        self._step(learn)


class HasLearnCB(Callback):
    def before_fit(self, learn):
        self.learn = learn

    def after_fit(self, learn):
        self.learn = None


class RecorderCB(Callback):
    def __init__(self, **d):
        self.d = d

    def before_fit(self, learn):
        self.recs = {k: [] for k in self.d}
        self.pg = learn.opt.param_groups[0]

    def after_batch(self, learn):
        if not learn.training:
            return
        for k, v in self.d.items():
            self.recs[k].append(v(self))

    def plot(self):
        for k, v in self.recs.items():
            plt.plot(v, label=k)
            plt.legend()
            plt.show()


class EpochSchedCB(BaseSchedCB):
    def after_epoch(self, learn):
        self._step(learn)


class MixedPrecision(TrainCB):
    order = DeviceCB.order + 10

    def before_fit(self, learn):
        self.scaler = torch.amp.GradScaler()

    def before_batch(self, learn):
        self.autocast = torch.autocast("cuda", dtype=torch.float16)
        self.autocast.__enter__()

    def after_loss(self, learn):
        self.autocast.__exit__(None, None, None)

    def backward(self, learn):
        self.scaler.scale(learn.loss).backward()

    def step(self, learn):
        self.scaler.step(learn.opt)
        self.scaler.update()


class AccelerateCB(TrainCB):
    order = DeviceCB.order + 10

    def __init__(self, n_inp=1, mixed_precision="fp16"):
        super().__init__(n_inp=n_inp)
        self.acc = Accelerator(mixed_precision=mixed_precision)

    def before_fit(self, learn):
        learn.model, learn.opt, learn.dls.train, learn.dls.valid = self.acc.prepare(
            learn.model, learn.opt, learn.dls.train, learn.dls.valid
        )

    def backward(self, learn):
        self.acc.backward(learn.loss)
