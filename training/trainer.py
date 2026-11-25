from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm


@dataclass
class TrainerConfig:
    lr: float = 0.0015
    max_epochs: int = 2
    device: torch.device | str | None = None
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    show_progress: bool = True
    save_checkpoints: bool = True

    def resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CVAETrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: Iterable[Any],
        test_dataloader: Iterable[Any],
        active_features: list[str],
        save_path: str | Path,
        experiment_id: str,
        trainer_config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.active_features = active_features
        self.experiment_id = experiment_id
        self.trainer_config = trainer_config or TrainerConfig()
        self.device = self.trainer_config.resolve_device()

        self.experiment_dir = Path(save_path) / experiment_id
        self.checkpoint_path = self.experiment_dir / 'best.pt'
        self.summary_path = self.experiment_dir / 'summary.json'

        self.best_epoch: int | None = None
        self.best_test_rec: float | None = None
        self.epoch_records: list[dict[str, Any]] = []

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)
        self.optimizer = self.trainer_config.optimizer_cls(
            self.model.parameters(),
            lr=self.trainer_config.lr,
            **self.trainer_config.optimizer_kwargs,
        )

    def _model_kwargs(self, batch: Any) -> dict[str, Any]:
        return {
            'pixel_values': batch.pixel_values.to(self.device),
            'feature_ids': {
                feature: getattr(batch.factors, feature).to(self.device)
                for feature in self.active_features
            },
        }

    def _iter_batches(self, dataloader: Iterable[Any], phase: str, epoch: int) -> Iterable[Any]:
        if not self.trainer_config.show_progress:
            return dataloader
        return tqdm(dataloader, desc=f'{phase} | Epoch {epoch + 1}')

    @staticmethod
    def _format_metrics(metrics: dict[str, float]) -> str:
        return ', '.join(f'{name}={value:.4f}' for name, value in metrics.items())

    @staticmethod
    def _progress_postfix(metrics: dict[str, list[float]]) -> dict[str, float]:
        display_names = {
            'loss': 'loss',
            'rec': 'rec_loss',
            'kld': 'kld_loss',
        }
        return {
            display_names[name]: sum(values) / len(values)
            for name, values in metrics.items()
            if values
        }

    @staticmethod
    def _record_metric(metrics: dict[str, list[float]], name: str, value: Any) -> None:
        metrics[name].append(float(value.detach() if isinstance(value, torch.Tensor) else value))

    def _run_epoch(self, dataloader: Iterable[Any], *, epoch: int, train: bool) -> dict[str, float]:
        self.model.train(mode=train)
        metrics: dict[str, list[float]] = defaultdict(list)
        phase = 'Train' if train else 'Test'

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            progress = self._iter_batches(dataloader, phase, epoch)
            for batch in progress:
                output = self.model(**self._model_kwargs(batch))

                if train:
                    self.optimizer.zero_grad()
                    output.loss.backward()
                    self.optimizer.step()

                self._record_metric(metrics, 'loss', output.loss)
                self._record_metric(metrics, 'rec', output.rec_loss)
                self._record_metric(metrics, 'kld', output.kld_loss)
                if hasattr(progress, 'set_postfix'):
                    progress.set_postfix(self._progress_postfix(metrics))

        return {
            name: sum(values) / len(values)
            for name, values in metrics.items()
        }

    def _save_best_checkpoint(self, *, epoch: int, test_rec: float) -> None:
        if self.best_test_rec is not None and test_rec >= self.best_test_rec:
            return

        self.best_epoch = epoch + 1
        self.best_test_rec = test_rec
        if not self.trainer_config.save_checkpoints:
            return
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def _write_summary(self) -> None:
        summary = {
            'experiment_id': self.experiment_id,
            'max_epochs': self.trainer_config.max_epochs,
            'best_epoch': self.best_epoch,
            'best_test_rec': self.best_test_rec,
            'checkpoint_path': str(self.checkpoint_path) if self.trainer_config.save_checkpoints else None,
            'epochs': self.epoch_records,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding='utf-8',
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        return self._run_epoch(self.train_dataloader, epoch=epoch, train=True)

    def evaluate(self, epoch: int) -> dict[str, float]:
        return self._run_epoch(self.test_dataloader, epoch=epoch, train=False)

    def fit(self) -> dict[str, list[float]]:
        history: dict[str, list[float]] = defaultdict(list)

        print(f'Starting training for {self.trainer_config.max_epochs} epoch(s) on {self.device}')
        for epoch in range(self.trainer_config.max_epochs):
            train_metrics = self.train_epoch(epoch)
            test_metrics = self.evaluate(epoch)

            for name, value in train_metrics.items():
                history[f'train_{name}'].append(value)
            for name, value in test_metrics.items():
                history[f'test_{name}'].append(value)

            self._save_best_checkpoint(epoch=epoch, test_rec=test_metrics['rec'])
            self.epoch_records.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'test': test_metrics,
            })
            self._write_summary()

            train_summary = self._format_metrics({
                f'train_{name}': value
                for name, value in train_metrics.items()
            })
            test_summary = self._format_metrics({
                f'test_{name}': value
                for name, value in test_metrics.items()
            })
            print(f'Epoch {epoch + 1}/{self.trainer_config.max_epochs} | {train_summary} | {test_summary}')

        return dict(history)
