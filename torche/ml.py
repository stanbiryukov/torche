import gc
import os
import time
from functools import partial

import mlflow
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch.utils.data import DataLoader


def init_mlflow():
    """
    Detect if running on databricks, and if so, set to their UI context.
    Else keep tracking URI as local files.
    """
    if "SPARK_HOME" in os.environ:
        if "databricks" in os.environ["SPARK_HOME"]:
            mlflow.set_tracking_uri("databricks")


class ExpMAStoppingCriterion:
    r"""Exponential moving average stopping criterion from BoTorch.
    # https://github.com/pytorch/botorch/blob/master/botorch/optim/stopping.py
    Computes an exponentially weighted moving average over window length `n_window`
    and checks whether the relative decrease in this moving average between steps
    is less than a provided tolerance level. That is, in iteration `i`, it computes
        v[i,j] := fvals[i - n_window + j] * w[j]
    for all `j = 0, ..., n_window`, where `w[j] = exp(-eta * (1 - j / n_window))`.
    Letting `ma[i] := sum_j(v[i,j])`, the criterion evaluates to `True` whenever
        (ma[i-1] - ma[i]) / abs(ma[i-1]) < rel_tol (if minimize=True)
        (ma[i] - ma[i-1]) / abs(ma[i-1]) < rel_tol (if minimize=False)
    """

    def __init__(
        self,
        maxiter: int = 10000,
        minimize: bool = True,
        n_window: int = 10,
        eta: float = 1.0,
        rel_tol: float = 1e-5,
    ) -> None:
        r"""Exponential moving average stopping criterion.
        Args:
            maxiter: Maximum number of iterations.
            minimize: If True, assume minimization.
            n_window: The size of the exponential moving average window.
            eta: The exponential decay factor in the weights.
            rel_tol: Relative tolerance for termination.
        """
        self.maxiter = maxiter
        self.minimize = minimize
        self.n_window = n_window
        self.rel_tol = rel_tol
        self.iter = 0
        weights = torch.exp(torch.linspace(-eta, 0, self.n_window))
        self.weights = weights / weights.sum()
        self._prev_fvals = None

    def evaluate(self, fvals: torch.as_tensor) -> bool:
        r"""Evaluate the stopping criterion.
        Args:
            fvals: tensor containing function values for the current iteration. If
                `fvals` contains more than one element, then the stopping criterion is
                evaluated element-wise and True is returned if the stopping criterion is
                true for all elements.
        TODO: add support for utilizing gradient information
        Returns:
            Stopping indicator (if True, stop the optimziation).
        """
        self.iter += 1
        if self.iter == self.maxiter:
            return True

        if self._prev_fvals is None:
            self._prev_fvals = fvals.unsqueeze(0)
        else:
            self._prev_fvals = torch.cat(
                [self._prev_fvals[-self.n_window :], fvals.unsqueeze(0)]
            )

        if self._prev_fvals.size(0) < self.n_window + 1:
            return False

        weights = self.weights
        weights = weights.to(fvals)
        if self._prev_fvals.ndim > 1:
            weights = weights.unsqueeze(-1)

        # TODO: Update the exp moving average efficiently
        prev_ma = (self._prev_fvals[:-1] * weights).sum(dim=0)
        ma = (self._prev_fvals[1:] * weights).sum(dim=0)
        # TODO: Handle approx. zero losses (normalize by min/max loss range)
        rel_delta = (prev_ma - ma) / prev_ma.abs()

        if not self.minimize:
            rel_delta = -rel_delta
        if torch.max(rel_delta) < self.rel_tol:
            return True

        return False


class NoneStep:
    """
    Dummy scheduler in case a real one isn't used.
    """

    def __init__(self):
        step = None

    def step(self):
        return None


class FFNN(torch.nn.Module):
    '''
    Torch Feed Forward Neural Network. Kaiming Linear Initialization.
    '''
    def __init__(self, input_dim, criterion, activation, layers):
        super(FFNN, self).__init__()
        layers = [input_dim] + list(layers)

        nn_stack = []
        for i, (dim_in, hidden_units) in enumerate(zip(layers[:-1], layers[1:])):
            ll = torch.nn.Linear(dim_in, hidden_units)
            nn_stack.append([ll, activation()])

        nn_stack.append([torch.nn.Linear(hidden_units, 1)])

        if any(z in criterion.__class__.__name__ for z in ["BCE"]):
            nn_stack.append([torch.nn.Sigmoid()])

        self.model = torch.nn.Sequential(*[l for x in nn_stack for l in x])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.model(x)


class Net(BaseEstimator, ClassifierMixin):
    '''
    Scikit-learn friendly Neural Network regressor and classifier.
    '''
    def __init__(
        self,
        x_scaler=StandardScaler(),
        y_scaler=StandardScaler(),
        random_state=581,
        activation=torch.nn.ReLU,
        hidden_layer_sizes=(
            256,
            512,
            512,
            64,
        ),
        net=FFNN,
        optimizer=partial(torch.optim.AdamW, lr=0.001),
        criterion=torch.nn.MSELoss(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        scheduler=None,
        verbose=False,
        l2_reg=True,
        l1_reg=False,
        max_iter=1000,
    ):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.random_state = random_state
        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.verbose = verbose
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.max_iter = max_iter

    def _to_tensor(self, tensor, dtype=torch.FloatTensor):
        return torch.as_tensor(tensor).to(self.device)

    def _setfit(self, random_state, X, y):
        self.set_seed(random_state)
        self.y = self._to_tensor(
            self.y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        )
        self.X = self._to_tensor(self.x_scaler.fit_transform(X).astype(np.float32))
        self.data_dim = self.X.shape[1]
        self.classes_ = np.unique(self.y)
        self.model = self.net(
            input_dim=self.data_dim,
            criterion=self.criterion,
            activation=self.activation,
            layers=self.hidden_layer_sizes,
        )
        self.model = self.model.to(self.device)
        self.optimizer = self.optimizer(params=list(self.model.parameters()))
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer)
        else:
            self.scheduler = NoneStep()

    def _fit(self):
        self.model.train()
        self.stopping_criterion = ExpMAStoppingCriterion()
        stop = False
        i = 0

        def closure():
            start = time.time()
            self.optimizer.zero_grad()
            try:  # catch any linear algebra errors
                output = self.model(self.X)
            except RuntimeError as e:
                if "singular" in e.args[0]:
                    return torch.as_tensor(np.nan)
                else:
                    raise e
            reg = 0
            if self.l2_reg:
                reg_ = torch.sum(self.model.model[0].weight ** 2) ** 0.5
                reg += reg_.detach().cpu()
            if self.l1_reg:
                reg_ = torch.sum(torch.abs(self.model.model[0].weight))
                reg += reg_.detach().cpu()

            loss = self.criterion(output, self.y)
            if any([self.l2_reg, self.l1_reg]):
                loss += reg

            if self.verbose:
                if i % 25 == 0:
                    print(
                        "Iter %d - Loss: %.3f - Took: %.3f [s]"
                        % (i, loss.item(), time.time() - start)
                    )
            loss.backward()
            return loss

        while (not stop) & (i < self.max_iter):

            loss = self.optimizer.step(closure)
            stop = self.stopping_criterion.evaluate(fvals=loss.detach().cpu())
            if "ReduceLROnPlateau" in self.scheduler.__class__.__name__:
                self.scheduler.step(loss.detach().cpu())
            else:
                self.scheduler.step()
            i += 1

    def fit(self, X, y):
        if not hasattr(self, "model"):
            """
            Set fit if there is no model object already
            """
            if self.verbose:
                print("Initializing model")
            self._setfit(random_state=self.random_state, X=X, y=y)
        self._fit()
        self.X = None
        self.y = None
        gc.collect()
        torch.cuda.empty_cache()

    def _predict(self, X):
        torch.cuda.empty_cache()
        self.set_seed(self.random_state)
        X = self._to_tensor(self.x_scaler.transform(X).astype(np.float32))
        covs_loader = DataLoader(X, batch_size=1024, shuffle=False)
        self.model.eval()
        ar = []

        with torch.no_grad():
            for data in covs_loader:
                preds = self.model(data.to(self.device)).detach().cpu()
                ar.append(preds)
        # now concat and rescale
        ar = self.y_scaler.inverse_transform(
            torch.cat(ar, dim=0).cpu().numpy().reshape(-1, 1)
        )
        return np.concatenate(ar, axis=0)

    def predict(self, X):
        hat = self._predict(X)
        if "bce" in type(self.criterion).__name__.lower():
            hat = np.where(hat > 0.5, 1, 0)
        return hat

    def predict_proba(self, X):
        hat = self._predict(X)
        return np.column_stack([1 - hat, hat])

    def set_seed(self, seed):
        import random

        import numpy as np
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class NNCV(BaseEstimator, ClassifierMixin):
    """
    Run NN parameter selection with joblib backend - faster than spark for small/medium sized datasets.
    Track best model with MLFlow.
    """

    def __init__(
        self,
        problem,
        random_state=2910,
        scoring=None,
        cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=300),
        ml_flow_logging=True,
    ):
        self.random_state = random_state
        self.get_random_state()
        self.cv = cv
        self.scoring = scoring if scoring is not None else self.get_scorer(problem)
        self.param_grid = self.get_param_grid(problem)
        self.ml_flow_logging = ml_flow_logging
        init_mlflow()

    def get_random_state(self):
        self._rstate = np.random.RandomState(self.random_state)

    def get_nn_arch(self):
        import itertools

        li0 = 512
        li1 = 256
        li2 = 128
        li3 = 64
        li = [li0, li1, li2, li3]
        nn_arch = []
        for i in range(len(li)):
            nn_arch.extend(list(itertools.product(li, repeat=i + 1)))
        return nn_arch

    def gen_id(self):
        import uuid

        fid = uuid.uuid4().hex
        return fid

    def fit(self, X, y):
        mlcv = GridSearchCV(
            Net(),
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            verbose=False,
            n_jobs=-1,
            refit=False,
            error_score=-999.0e9,
        )
        mlcv.fit(X, y)
        self.best_estimator = Net(**mlcv.best_params_)
        self.run_id = self.gen_id()
        self.best_estimator_cv = cross_val_score(
            self.best_estimator, X=X, y=y, n_jobs=-1, scoring=self.scoring, cv=self.cv
        )
        if self.ml_flow_logging in [1]:
            with mlflow.start_run(run_name=f"best_{self.run_id}"):
                self.best_estimator.fit(X, y)
                for k, v in self.best_estimator.get_params().items():
                    mlflow.log_param(k, v)
                mlflow.log_metric(f"cv_{self.scoring}", np.mean(self.best_estimator_cv))
                in_sample = get_scorer(self.scoring)(self.best_estimator, X, y)
                mlflow.log_metric(f"training_{self.scoring}", in_sample)
        else:
            self.best_estimator.fit(X, y)

    def predict(self, X):
        return self.best_estimator.predict(X)

    def predict_proba(self, X):
        return self.best_estimator.predict_proba(X)

    def get_scorer(self, problem):
        if "reg" in problem.lower():
            scorer = "neg_root_mean_squared_error"
        else:
            scorer = "neg_log_loss"
        return scorer

    def get_param_grid(self, problem):
        if "reg" in problem.lower():
            search_space = {
                "hidden_layer_sizes": self.get_nn_arch(),
                "random_state": [self.random_state],
                "activation": [torch.nn.ReLU, torch.nn.Tanh],
            }
        else:
            search_space = {
                "hidden_layer_sizes": self.get_nn_arch(),
                "criterion": [torch.nn.BCELoss()],
                "y_scaler": [FunctionTransformer(validate=True)],
                "random_state": [self.random_state],
                "activation": [torch.nn.ReLU, torch.nn.Tanh],
            }
        return search_space
