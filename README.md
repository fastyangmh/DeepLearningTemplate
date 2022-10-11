# DeepLearningTemplate

This repository is my deep learning template. It will be a submodule in many of my deep learning repositories.

## Installation

```bash
pip install -r requirements.txt
```

## Notice

In [tuning.py](tuning.py#L1), there is a reproducibility problem in NNI which is unable to reproduce the same results as the trials

In [dataset.py](dataset.py#L323), the MyBreastCancerDataset dataset only contains training and validation datasets and the ratio is 8:2

## TODO

### General

- [ ] add create_datamodule function in data_preparation.py
- [ ] explainable AI via GradCAM or Captum
- [ ] prediction via Restful API
- [ ] convert model to ONNX model
- [ ] prediction via ONNX model
- [ ] make decrease_samples function more flexible and general
- [ ] automatically detect the number of classes in target_transforms_config while parse_transforms function
- [ ] evaluate TorchData
- [ ] add UnsupervisedModel class in model.py
- [ ] add selfdefined_models.py

### Series task

- [ ] support multiple file extensions in MySeriesFolder class
- [ ] add TorchArrow library for series task

### Audio task

- [ ] add torch-audiomentations transforms for audio task

### Video task

- [ ] add video dataset for video task
