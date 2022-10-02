#import
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, DeviceStatsMonitor, RichProgressBar
from typing import Union, List, Callable
import torch
import yaml
from os.path import join


#class
class Trainer:
    def __init__(self, project_parameters: argparse.Namespace,
                 datamodule_function: Callable,
                 model_function: Callable) -> None:
        pl.seed_everything(seed=project_parameters.random_seed)
        self.trainer = self.create_trainer(
            early_stopping=project_parameters.early_stopping,
            patience=project_parameters.patience,
            accelerator=project_parameters.accelerator,
            devices=project_parameters.devices,
            default_root_dir=project_parameters.default_root_dir,
            precision=project_parameters.precision,
            max_epochs=project_parameters.max_epochs,
        )
        self.datamodule = datamodule_function(project_parameters)
        self.model = model_function(project_parameters)
        self.project_parameters = project_parameters

    def create_trainer(self, early_stopping: bool, patience: int,
                       accelerator: str, default_root_dir: str,
                       devices: Union[List[int], str,
                                      int], precision: int, max_epochs: int):
        callbacks = [
            LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.4f}',
                            monitor='val_loss',
                            mode='min',
                            save_last=True,
                            save_weights_only=True),
            DeviceStatsMonitor(cpu_stats=True),
            RichProgressBar(leave=True)
        ]
        if early_stopping:
            callbacks.append(
                EarlyStopping(monitor='val_loss',
                              patience=patience,
                              mode='min'))
        if accelerator == 'gpu' and torch.cuda.is_available():
            accelerator = 'gpu'
        else:
            accelerator = 'cpu'
        return pl.Trainer(
            accelerator=accelerator,
            callbacks=callbacks,
            check_val_every_n_epoch=1,
            default_root_dir=default_root_dir,
            deterministic=
            'warn',  #Set to "warn" to use deterministic algorithms whenever possible, throwing warnings on operations that don’t support deterministic mode (requires PyTorch 1.11+).
            devices=devices,
            precision=precision,
            max_epochs=max_epochs)

    def __call__(self):
        self.trainer.fit(model=self.model, datamodule=self.datamodule)
        self.datamodule.setup(stage='test')
        dataloaders_dict = {
            'train': self.datamodule.train_dataloader(),
            'val': self.datamodule.val_dataloader(),
            'test': self.datamodule.test_dataloader()
        }
        result = {'trainer': self.trainer, 'model': self.model}
        for stage, dataloader in dataloaders_dict.items():
            result[stage] = self.trainer.test(dataloaders=dataloader,
                                              ckpt_path='best')[0]
        #save project_parameters to default_root_dir
        with open(join(self.trainer.logger.log_dir, 'config.yaml'),
                  'w') as stream:
            yaml.dump(data=vars(self.project_parameters), stream=stream)
        return result