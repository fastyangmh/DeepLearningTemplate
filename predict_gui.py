# import
from __future__ import annotations
from . import Predictor
import argparse
from typing import Callable, Optional, List, Any
import gradio as gr
from gradio.components import Component


# class
class PredictorGUI:
    def __init__(
            self,
            project_parameters: argparse.Namespace,
            loader: Callable,
            gradio_inputs: Optional[str | Component | List[str | Component]],
            gradio_outputs: Optional[str | Component | List[str | Component]],
            examples: Optional[List[Any] | List[List[Any]] | str] = None
    ) -> None:
        self.predictor = Predictor(project_parameters=project_parameters,
                                   loader=loader)
        self.gui = gr.Interface(fn=self.inference,
                                inputs=gradio_inputs,
                                outputs=gradio_outputs,
                                examples=examples,
                                cache_examples=True,
                                live=True,
                                interpretation='default')
        self.classes = project_parameters.classes

    def inference(self, inputs):
        prediction = self.predictor(
            inputs=inputs)  #prediction dimension is (1, num_classes)
        prediction = prediction[0]
        return {cls: proba for cls, proba in zip(self.classes, prediction)}

    def __call__(self):
        self.gui.launch(inbrowser=True, share=True)
