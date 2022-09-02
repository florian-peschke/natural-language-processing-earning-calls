import os
import re
import typing as t
from collections import defaultdict
from typing import List, Union

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from attr import define, field
from numpy.random._generator import Generator
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typeguard import typechecked

from ecc import ENTITY, log_wandb, SEED, wandb_project
from ecc.torch import LABEL_QA1, LABEL_QA2, LABEL_QA3, LABEL_TOTAL_ACCURACY
from ecc.torch.dataset import DatasetPipeline
from ecc.utils import create_dir_if_missing, frame_to_latex, save_optuna_plot, to_pickle, to_yaml

CHOICE_NUMBER_OF_UNITS: t.Tuple[int] = tuple(int(2**e) for e in np.arange(4, 10))
CHOICE_DROP_OUT_RATES: t.Tuple[float] = tuple(float(2.0**e / 100) for e in np.arange(6))
METRIC_NAME_FORMAT: str = "{prefix}_accuracy_{label}"
LOSS_NAME_FORMAT: str = "{prefix}_bce_with_logits_loss_{label}"

PREFIX_TRAIN: str = "train"
PREFIX_VAL: str = "val"
PREFIX_TEST: str = "test"

EARLY_STOPPING_PATIENCE: int = 5
ReduceLROnPlateau_PATIENCE: int = np.max([EARLY_STOPPING_PATIENCE // 2, 1])


def yield_loss(predictions: t.Dict[str, torch.Tensor], labels: t.Dict[str, torch.Tensor]) -> t.Iterable[torch.Tensor]:
    pred: torch.Tensor
    true: torch.Tensor
    for pred, true in zip(predictions.values(), labels.values()):
        yield torch.nn.BCEWithLogitsLoss()(pred.squeeze(1), true.squeeze(1))


row_wise_eq: t.Callable = lambda pred, target: torch.eq(torch.round(pred).type(target.type()), target)
reduced_binary_accuracy: t.Callable = (
    lambda pred, target: row_wise_eq(pred=pred, target=target).prod(dim=1).double().mean()
)


def init_weights_and_biases(layer: t.Any) -> None:
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)


# noinspection StrFormat
def multi_label_class_accuracy(
    predictions: t.Dict[str, torch.Tensor], labels: t.Dict[str, torch.Tensor], prefix: str
) -> t.Dict[str, torch.Tensor]:
    qa1_accuracy: torch.Tensor = reduced_binary_accuracy(
        pred=nn.Sigmoid()(predictions[LABEL_QA1]).squeeze(1), target=labels[LABEL_QA1].squeeze(1).int()
    )
    qa2_accuracy: torch.Tensor = reduced_binary_accuracy(
        pred=nn.Sigmoid()(predictions[LABEL_QA2]).squeeze(1), target=labels[LABEL_QA2].squeeze(1).int()
    )
    qa3_accuracy: torch.Tensor = reduced_binary_accuracy(
        pred=nn.Sigmoid()(predictions[LABEL_QA3]).squeeze(1), target=labels[LABEL_QA3].squeeze(1).int()
    )
    return {
        METRIC_NAME_FORMAT.format(prefix=prefix, label=LABEL_TOTAL_ACCURACY): torch.mean(
            torch.stack([qa1_accuracy, qa2_accuracy, qa3_accuracy])
        ),
        METRIC_NAME_FORMAT.format(prefix=prefix, label=LABEL_QA1.lower()): qa1_accuracy,
        METRIC_NAME_FORMAT.format(prefix=prefix, label=LABEL_QA2.lower()): qa2_accuracy,
        METRIC_NAME_FORMAT.format(prefix=prefix, label=LABEL_QA3.lower()): qa3_accuracy,
    }


# noinspection StrFormat
class MultiClassMultiLabelClassifier(pl.LightningModule):
    config: t.Dict[str, t.Any]
    trial_qa1: t.Union[optuna.Trial, FrozenTrial]
    trial_qa2: t.Union[optuna.Trial, FrozenTrial]
    trial_qa3: t.Union[optuna.Trial, FrozenTrial]

    def __init__(
        self,
        trial_qa1: t.Union[optuna.Trial, FrozenTrial],
        trial_qa2: t.Union[optuna.Trial, FrozenTrial],
        trial_qa3: t.Union[optuna.Trial, FrozenTrial],
        output_units: t.Dict[str, int],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.trial_qa1 = trial_qa1
        self.trial_qa2 = trial_qa2
        self.trial_qa3 = trial_qa3

        # hidden layer architecture for three different models (a separate model for each qa type)
        self.qa1: nn.Sequential = nn.Sequential(
            nn.LazyLinear(self.trial_qa1.suggest_categorical("input_layer_output_dim_qa1", CHOICE_NUMBER_OF_UNITS)),
            nn.ReLU(),
        )
        self.qa2: nn.Sequential = nn.Sequential(
            nn.LazyLinear(self.trial_qa2.suggest_categorical("input_layer_output_dim_qa2", CHOICE_NUMBER_OF_UNITS)),
            nn.ReLU(),
        )
        self.qa3: nn.Sequential = nn.Sequential(
            nn.LazyLinear(self.trial_qa3.suggest_categorical("input_layer_output_dim_qa3", CHOICE_NUMBER_OF_UNITS)),
            nn.ReLU(),
        )

        # draw the number of hidden layers
        n_hidden_layer_blocks_qa1: int = self.trial_qa1.suggest_int(
            "n_hidden_layer_blocks_qa1", low=1, high=10, log=True
        )
        n_hidden_layer_blocks_qa2: int = self.trial_qa2.suggest_int(
            "n_hidden_layer_blocks_qa2", low=1, high=10, log=True
        )
        n_hidden_layer_blocks_qa3: int = self.trial_qa3.suggest_int(
            "n_hidden_layer_blocks_qa3", low=1, high=10, log=True
        )

        # create hidden layer blocks
        n_hidden_layers: int
        sequential: nn.Sequential
        name: str
        for n_hidden_layers, sequential, name, trial in zip(
            [n_hidden_layer_blocks_qa1, n_hidden_layer_blocks_qa2, n_hidden_layer_blocks_qa3],
            [self.qa1, self.qa2, self.qa3],
            [LABEL_QA1, LABEL_QA2, LABEL_QA3],
            [self.trial_qa1, self.trial_qa2, self.trial_qa3],
        ):
            for i in np.arange(1, n_hidden_layers + 1):
                # add the main hidden (linear) layer
                sequential.add_module(
                    f"hidden_layer_{i}",
                    nn.LazyLinear(
                        trial.suggest_categorical(f"hidden_layer_{i}_output_dim_{name}", CHOICE_NUMBER_OF_UNITS)
                    ),
                )
                # add the respective activation function
                sequential.add_module(f"hidden_layer_{i}_activation", nn.ReLU())
                # add the drop out layer
                sequential.add_module(
                    f"hidden_layer_{i}_dropout",
                    nn.Dropout(trial.suggest_categorical(f"hidden_layer_{i}_dropout_{name}", CHOICE_DROP_OUT_RATES)),
                )

            # add the output layer
            sequential.add_module("output_layer", nn.LazyLinear(output_units[name]))
            # info: no (final) activation function necessary as the loss is 'BCEWithLogitsLoss', which takes the
            # logits as input and is more numerically stable than using a plain Sigmoid followed by a BCELoss
            # (see https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

            # turn automatic optimization off
            self.automatic_optimization = False

    def init_lazy_layers(self, batch: DataLoader) -> None:
        """
        Init weights and biases as well as the dimensions of the lazy layers.
        """
        # get just x and ignore y
        x: torch.Tensor
        x, _ = batch

        # one forward pass to init lazy layers parameters (converts a LazyLinear to a normal Linear layer)
        _ = self.forward(x)

    def init_parameters(self) -> None:
        # init weights and biases
        sequential: nn.Sequential
        for sequential in [self.qa1, self.qa2, self.qa3]:
            sequential.apply(init_weights_and_biases)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> t.Dict[str, torch.Tensor]:
        inputs: torch.Tensor = x.to(torch.float)
        return {
            LABEL_QA1: self.qa1(inputs),
            LABEL_QA2: self.qa2(inputs),
            LABEL_QA3: self.qa3(inputs),
        }

    def training_step(
        self,
        batch: DataLoader,
        batch_idx: t.Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        loss: torch.Tensor
        accuracies: t.Dict[str, torch.Tensor]
        loss, accuracies = self._step(batch=batch, prefix=PREFIX_TRAIN, is_training=True)
        return loss

    def validation_step(
        self, batch: DataLoader, batch_idx: t.Optional[int] = None, dataloader_idx: t.Optional[int] = None, **kwargs
    ) -> t.Dict[str, torch.Tensor]:
        loss: torch.Tensor
        accuracies: t.Dict[str, torch.Tensor]
        loss, accuracies = self._step(batch=batch, prefix=PREFIX_VAL)
        return accuracies

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        metric_label: str
        scheduler: ReduceLROnPlateau
        for metric_label, scheduler in zip([LABEL_QA1, LABEL_QA2, LABEL_QA3], self.lr_schedulers()):
            scheduler.step(
                self.trainer.callback_metrics[METRIC_NAME_FORMAT.format(prefix=PREFIX_VAL, label=metric_label.lower())]
            )

    def test_step(
        self, batch: DataLoader, batch_idx: t.Optional[int] = None, dataloader_id: t.Optional[int] = None, **kwargs
    ) -> t.Dict[str, torch.Tensor]:
        loss: torch.Tensor
        accuracies: t.Dict[str, torch.Tensor]
        loss, accuracies = self._step(batch=batch, prefix=PREFIX_TEST)
        return accuracies

    def predict_step(self, batch: DataLoader, batch_idx: int, dataloader_idx: int = 0) -> t.Dict[str, np.ndarray]:
        # get x and y
        x: torch.Tensor
        y: t.Dict[str, torch.Tensor]
        x, y = batch
        y = {name: tensor.squeeze(1) for name, tensor in y.items()}

        # get predictions
        predictions: t.Dict[str, torch.Tensor] = self.forward(x)

        # get row wise eq
        row_wise_equal: t.Dict[str, torch.Tensor] = {
            f"{name}_row_wise_equal": row_wise_eq(pred=pred.squeeze(1), target=target.squeeze(1)).int()
            for name, pred, target in zip(predictions.keys(), predictions.values(), y.values())
        }

        # get accuracies
        accuracies: t.Dict[str, torch.Tensor] = multi_label_class_accuracy(
            predictions=predictions, labels=y, prefix=PREFIX_TEST
        )

        # update dictionary
        predictions: t.Dict[str, torch.Tensor] = {
            f"{name}_pred": torch.round(nn.Sigmoid()(tensor)).int() for name, tensor in predictions.items()
        }
        predictions.update({f"{name}_true": tensor.int() for name, tensor in y.items()})
        predictions.update(row_wise_equal)
        predictions.update(accuracies)

        return {
            name: predictions[name].cpu().numpy()
            for name in sorted(
                predictions.keys(), key=lambda text: re.search("(?P<type>.*)_.*", text).groupdict()["type"]
            )
        }

    def _step(
        self, batch: DataLoader, prefix: str, is_training: bool = False
    ) -> t.Tuple[torch.Tensor, t.Dict[str, torch.Tensor]]:

        # get x and y
        x: torch.Tensor
        y: t.Dict[str, torch.Tensor]
        x, y = batch
        y = {name: tensor.squeeze(1) for name, tensor in y.items()}

        # get the optimizers
        opt_qa1: torch.optim.Optimizer
        opt_qa2: torch.optim.Optimizer
        opt_qa3: torch.optim.Optimizer
        opt_qa1, opt_qa2, opt_qa3 = self.optimizers()

        # get predictions
        predictions: t.Dict[str, torch.Tensor] = self.forward(x)

        # calculate losses
        loss_qa1: torch.Tensor
        loss_qa2: torch.Tensor
        loss_qa3: torch.Tensor
        loss_qa1, loss_qa2, loss_qa3 = tuple(list(yield_loss(predictions=predictions, labels=y)))

        # calculate accuracies
        accuracies: t.Dict[str, torch.Tensor] = multi_label_class_accuracy(
            predictions=predictions, labels=y, prefix=prefix
        )

        # if training, back-propagate and update optimizers
        if is_training:
            # run backpropagation and update the optimizer
            for loss, optimizer in zip([loss_qa1, loss_qa2, loss_qa3], [opt_qa1, opt_qa2, opt_qa3]):
                optimizer.zero_grad()
                self.manual_backward(loss)
                optimizer.step()

        # log results
        self.log_dict(
            dict(
                **accuracies,
                **{
                    LOSS_NAME_FORMAT.format(prefix=prefix, label=LABEL_TOTAL_ACCURACY): torch.sum(
                        torch.stack([loss_qa1, loss_qa2, loss_qa3])
                    ),
                    LOSS_NAME_FORMAT.format(prefix=prefix, label=LABEL_QA1.lower()): loss_qa1,
                    LOSS_NAME_FORMAT.format(prefix=prefix, label=LABEL_QA2.lower()): loss_qa2,
                    LOSS_NAME_FORMAT.format(prefix=prefix, label=LABEL_QA3.lower()): loss_qa3,
                },
            ),
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

        return torch.sum(torch.stack((loss_qa1, loss_qa2, loss_qa3))), accuracies

    def configure_optimizers(
        self,
    ) -> t.Tuple[t.List[torch.optim.Optimizer], t.List[ReduceLROnPlateau]]:
        qa1: torch.optim.Optimizer = torch.optim.Adam(
            self.qa1.parameters(), lr=self.trial_qa1.suggest_loguniform("lr_qa1", 0.001, 0.5)
        )
        qa2: torch.optim.Optimizer = torch.optim.Adam(
            self.qa2.parameters(), lr=self.trial_qa2.suggest_loguniform("lr_qa2", 0.001, 0.5)
        )
        qa3: torch.optim.Optimizer = torch.optim.Adam(
            self.qa3.parameters(), lr=self.trial_qa3.suggest_loguniform("lr_qa3", 0.001, 0.5)
        )
        # init learning rate schedulers to reduce the learning if the accuracies do not improve after fixed number of
        # epochs (patience)
        qa1_lr_scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            optimizer=qa1, mode="max", patience=ReduceLROnPlateau_PATIENCE
        )
        qa2_lr_scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            optimizer=qa2, mode="max", patience=ReduceLROnPlateau_PATIENCE
        )
        qa3_lr_scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            optimizer=qa3, mode="max", patience=ReduceLROnPlateau_PATIENCE
        )
        return [qa1, qa2, qa3], [qa1_lr_scheduler, qa2_lr_scheduler, qa3_lr_scheduler]


# noinspection StrFormat
@define
class NNRoutine:
    batch_size: int
    inputs: t.Dict[str, torch.Tensor]
    name: str
    name_postfix: str
    outputs: t.Dict[str, t.Dict[str, np.ndarray]]
    path: str
    scaling_range: t.Optional[t.Tuple[int, int]]
    shuffle: bool

    _dataset_pipeline: DatasetPipeline = field(init=False)

    _study_qa1: optuna.Study = field(init=False)
    _study_qa2: optuna.Study = field(init=False)
    _study_qa3: optuna.Study = field(init=False)

    _max_epochs_tuning: int = field(init=False)
    _max_epochs_evaluation: int = field(init=False)

    _trial_dict: defaultdict = field(init=False)

    @typechecked
    def __init__(
        self,
        inputs: t.Dict[str, torch.Tensor],
        outputs: t.Dict[str, t.Dict[str, np.ndarray]],
        batch_size: int,
        name: str,
        name_postfix: str,
        path: str,
        shuffle: bool = True,
        scaling_range: t.Optional[t.Tuple[int, int]] = (-1, 1),
    ) -> None:
        self.__attrs_init__(
            inputs=inputs,
            outputs=outputs,
            batch_size=batch_size,
            name=name.replace("/", "-"),
            name_postfix=name_postfix,
            path=path,
            shuffle=shuffle,
            scaling_range=scaling_range,
        )

    def __attrs_post_init__(self) -> None:
        self._dataset_pipeline = DatasetPipeline(
            inputs=self.inputs,
            outputs=self.outputs,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            scaling_range=self.scaling_range,
        )

    def tune(self, max_epochs_tuning: int, max_epochs_evaluation: int, n_trials: int) -> None:

        # init class parameters
        self._max_epochs_tuning = max_epochs_tuning
        self._max_epochs_evaluation = max_epochs_evaluation
        self._trial_dict = defaultdict(dict)

        # init random number generator
        rnd_generation: Generator = np.random.default_rng(SEED)

        self._study_qa1 = optuna.create_study(
            direction="maximize",
            study_name=f"{self.name.lower()} – {self.name_postfix.lower()} – {LABEL_QA1.lower()}",
            pruner=optuna.pruners.MedianPruner(),
            sampler=TPESampler(seed=rnd_generation.integers(2**16)),
        )
        self._study_qa2 = optuna.create_study(
            direction="maximize",
            study_name=f"{self.name.lower()} – {self.name_postfix.lower()} – {LABEL_QA2.lower()}",
            pruner=optuna.pruners.MedianPruner(),
            sampler=TPESampler(seed=rnd_generation.integers(2**16)),
        )
        self._study_qa3 = optuna.create_study(
            direction="maximize",
            study_name=f"{self.name.lower()} – {self.name_postfix.lower()} – {LABEL_QA3.lower()}",
            pruner=optuna.pruners.MedianPruner(),
            sampler=TPESampler(seed=rnd_generation.integers(2**16)),
        )

        # run tuning trials
        for i in range(n_trials):
            acc_all: float
            acc_qa1: float
            acc_qa2: float
            acc_qa3: float

            # get the trial objects
            trial_qa1: optuna.Trial = self._study_qa1.ask()
            trial_qa2: optuna.Trial = self._study_qa2.ask()
            trial_qa3: optuna.Trial = self._study_qa3.ask()

            # tune the model and set the accuracies
            acc_all, acc_qa1, acc_qa2, acc_qa3 = self._objective(
                trail_qa1=trial_qa1,
                trial_qa2=trial_qa2,
                trial_qa3=trial_qa3,
            )

            # updating optuna studies
            self._study_qa1.tell(trial=trial_qa1, values=acc_qa1, state=optuna.trial.TrialState.COMPLETE)
            self._study_qa2.tell(trial=trial_qa2, values=acc_qa2, state=optuna.trial.TrialState.COMPLETE)
            self._study_qa3.tell(trial=trial_qa3, values=acc_qa3, state=optuna.trial.TrialState.COMPLETE)

            # store trial results in self._trial_dict
            if log_wandb:
                trial: optuna.Trial
                label_name: str
                accuracy: float
                for trial, label_name, accuracy in zip(
                    [trial_qa1, trial_qa2, trial_qa3],
                    [LABEL_QA1, LABEL_QA2, LABEL_QA3],
                    [acc_qa1, acc_qa2, acc_qa3],
                ):
                    self._trial_dict[f"ml_tuning - {self.name} - {self.name_postfix} – {label_name}"].update(
                        {
                            i: dict(
                                **trial.params,
                                **{
                                    METRIC_NAME_FORMAT.format(prefix=PREFIX_VAL, label=LABEL_TOTAL_ACCURACY): acc_all,
                                    "accuracy": accuracy,
                                },
                            ),
                        }
                    )

        # log results (wandb)
        if log_wandb:
            for name, trial_dict in self._trial_dict.items():
                with wandb.init(
                    name=name,
                    project=wandb_project,
                    entity=ENTITY,
                    group=f"ml_tuning - {self.name} - {self.name_postfix}",
                ):
                    for step, log_parameters in trial_dict.items():
                        wandb.log(log_parameters, step=step)

        # save results
        self._post_tuning()

    def _post_tuning(self) -> None:

        # fit again with the best hyperparameters and then evaluate
        metrics: t.Dict[str, torch.Tensor]
        model: MultiClassMultiLabelClassifier
        trainer: pl.Trainer
        metrics, model, trainer = self._eval_best_model()

        # predict
        predictions: t.List[t.Dict[str, np.ndarray]] = trainer.predict(model, datamodule=self._dataset_pipeline)

        # save predictions
        pred_frame: pd.DataFrame = pd.DataFrame(predictions)
        pred_frame.to_csv(os.path.join(self.path, "predictions.csv"))

        # save metrics as .csv
        frame: pd.DataFrame = (
            pd.Series(
                {
                    metric_name: metric_value.item()
                    for metric_name, metric_value in metrics.items()
                    if not bool(re.search(r".*_epoch", metric_name)) and bool(re.search(r"acc", metric_name))
                },
                name=self.name,
            )
            .to_frame()
            .T
        )
        frame.to_csv(os.path.join(self.path, "evaluation.csv"), index_label="model")
        frame_to_latex(frame=frame, path=self.path, name="evaluation")

        # save the model architecture as .txt
        with open(os.path.join(self.path, "best_model.txt"), "w") as f:
            print(model, file=f)

        # save tuning results
        study: optuna.Study
        name: str
        for study, name in zip([self._study_qa1, self._study_qa2, self._study_qa3], [LABEL_QA1, LABEL_QA2, LABEL_QA3]):
            path: str = create_dir_if_missing(os.path.join(self.path, name))
            # save hyper-parameters as .yaml
            to_yaml(
                dictionary=study.best_trial.params,
                filename=os.path.join(
                    path,
                    "hyperparameter.yaml",
                ),
            )

            # save the hyperparameter importance as .pdf
            save_optuna_plot(
                path=path,
                name="hyperparameter_importance",
                study=study,
                plot_fun=optuna.visualization.plot_param_importances,
            )

            # save the slice plot as .pdf
            save_optuna_plot(
                path=path,
                name="plot_slice",
                study=study,
                plot_fun=optuna.visualization.plot_slice,
            )

            # save study info as .csv
            study.trials_dataframe().to_csv(
                os.path.join(path, "study.csv"),
                index=True,
            )

            # save study as .pkl
            to_pickle(object_=study.trials, filename=os.path.join(path, "study_trials.pkl"))

        # finish wandb logging to save and upload the results
        if log_wandb:
            wandb.finish()

    def _eval_best_model(self) -> t.Tuple[t.Dict[str, torch.Tensor], MultiClassMultiLabelClassifier, pl.Trainer]:
        trainer: pl.Trainer
        model: MultiClassMultiLabelClassifier
        trainer, model = self._fit(
            trial_qa1=self._study_qa1.best_trial,
            trial_qa2=self._study_qa2.best_trial,
            trial_qa3=self._study_qa3.best_trial,
            max_epochs=self._max_epochs_tuning,
        )
        trainer.test(model, datamodule=self._dataset_pipeline)
        return trainer.callback_metrics, model, trainer

    def _fit(
        self,
        trial_qa1: t.Union[optuna.Trial, FrozenTrial],
        trial_qa2: t.Union[optuna.Trial, FrozenTrial],
        trial_qa3: t.Union[optuna.Trial, FrozenTrial],
        max_epochs: int,
    ) -> t.Tuple[pl.Trainer, MultiClassMultiLabelClassifier]:
        # set up Trainer
        trainer: pl.Trainer = pl.Trainer(
            logger=True,
            max_epochs=max_epochs,
            callbacks=[
                EarlyStopping(
                    monitor=METRIC_NAME_FORMAT.format(prefix=PREFIX_VAL, label=LABEL_TOTAL_ACCURACY),
                    min_delta=0.001,
                    mode="max",
                    patience=EARLY_STOPPING_PATIENCE,
                    verbose=True,
                ),
                # RichProgressBar(refresh_rate=8),
                LearningRateMonitor(logging_interval="step"),
            ],
            **dict(accelerator="gpu", devices=1) if torch.cuda.is_available() else dict(),
        )

        # init model
        model: MultiClassMultiLabelClassifier = MultiClassMultiLabelClassifier(
            trial_qa1=trial_qa1,
            trial_qa2=trial_qa2,
            trial_qa3=trial_qa3,
            output_units=self._dataset_pipeline.output_units,
        )

        # init lazy layers
        model.init_lazy_layers(next(iter(self._dataset_pipeline.training)))

        # init layer parameters
        model.init_parameters()

        # fit the model
        trainer.fit(
            model,
            datamodule=self._dataset_pipeline,
        )

        return trainer, model

    def _objective(
        self, trail_qa1: optuna.Trial, trial_qa2: optuna.Trial, trial_qa3: optuna.Trial
    ) -> t.Tuple[float, float, float, float]:
        metric_name_acc_all: str = METRIC_NAME_FORMAT.format(prefix=PREFIX_VAL, label=LABEL_TOTAL_ACCURACY)
        metric_name_acc_qa1: str = METRIC_NAME_FORMAT.format(prefix=PREFIX_VAL, label=LABEL_QA1.lower())
        metric_name_acc_qa2: str = METRIC_NAME_FORMAT.format(prefix=PREFIX_VAL, label=LABEL_QA2.lower())
        metric_name_acc_qa3: str = METRIC_NAME_FORMAT.format(prefix=PREFIX_VAL, label=LABEL_QA3.lower())

        # train the model with drawn hyperparameters
        trainer: pl.Trainer
        model: MultiClassMultiLabelClassifier
        trainer, model = self._fit(
            trial_qa1=trail_qa1, trial_qa2=trial_qa2, trial_qa3=trial_qa3, max_epochs=self._max_epochs_tuning
        )
        # return accuracies
        return (
            trainer.callback_metrics[metric_name_acc_all].item(),
            trainer.callback_metrics[metric_name_acc_qa1].item(),
            trainer.callback_metrics[metric_name_acc_qa2].item(),
            trainer.callback_metrics[metric_name_acc_qa3].item(),
        )
