# import
from __future__ import annotations
import argparse
from typing import Callable, Optional, List, Any, Union
import gradio as gr
from gradio.components import Component
import pytorch_lightning as pl
from abc import ABC, abstractmethod
import torch
try:
    from . import Predictor
except:
    from predict import Predictor


# class
class BasePredictorGUI(ABC):
    def __init__(
            self,
            project_parameters: argparse.Namespace,
            model: pl.LightningModule,
            loader: Callable,
            gradio_inputs: Optional[str | Component | List[str | Component]],
            gradio_outputs: Optional[str | Component | List[str | Component]],
            examples: Optional[List[Any] | List[List[Any]] | str] = None
    ) -> None:
        super().__init__()
        self.classes = project_parameters.classes
        self.predictor = Predictor(project_parameters=project_parameters,
                                   model=model,
                                   loader=loader)
        self.gradio_inputs = gradio_inputs
        self.gradio_outputs = gradio_outputs
        self.examples = examples

    @abstractmethod
    def inference(self, inputs: Union[str, torch.Tensor]):
        return NotImplementedError

    def __call__(self):
        self.gui = gr.Interface(fn=self.inference,
                                inputs=self.gradio_inputs,
                                outputs=self.gradio_outputs,
                                examples=self.examples,
                                cache_examples=True,
                                live=True,
                                interpretation='default')
        self.gui.launch(inbrowser=True, share=True)


class ClassificationPredictorGUI(BasePredictorGUI):
    def __init__(
            self,
            project_parameters: argparse.Namespace,
            model: pl.LightningModule,
            loader: Callable,
            gradio_inputs: Optional[str | Component | List[str | Component]],
            gradio_outputs: Optional[str | Component | List[str | Component]],
            examples: Optional[List[Any] | List[List[Any]] | str] = None
    ) -> None:
        super().__init__(project_parameters, model, loader, gradio_inputs,
                         gradio_outputs, examples)

    def inference(self, inputs: Union[str, torch.Tensor]):
        prediction = self.predictor(
            inputs=inputs)  #prediction dimension is (1, num_classes)
        prediction = prediction[0]
        return {cls: proba for cls, proba in zip(self.classes, prediction)}