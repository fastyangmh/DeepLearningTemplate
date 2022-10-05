#NOTE: There is a reproducibility problem in NNI which is unable to reproduce the same results as the trials

#import
import yaml
import argparse
from os import makedirs
from os.path import join, basename, realpath
import datetime
from copy import deepcopy
import subprocess
from typing import Any, Dict


#class
class Tuner:
    def __init__(self, project_parameters: argparse.Namespace) -> None:
        self.project_parameters = project_parameters
        self.nni_port = project_parameters.nni_port

    def set_config(self, project_parameters: argparse.Namespace):
        experimentWorkingDirectory = realpath(
            project_parameters.nni_config['experimentWorkingDirectory'])
        project_parameters.nni_config[
            'experimentWorkingDirectory'] = experimentWorkingDirectory
        makedirs(experimentWorkingDirectory, exist_ok=True)
        path = join(
            experimentWorkingDirectory,
            f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')

        #set config
        data = vars(deepcopy(project_parameters))
        data['mode'] = 'train'
        project_parameters_config_filepath = path + f'_{basename(data["config"])}'
        data['nni_config']['trialConcurrency'] = 1
        data['nni_config']['trialCodeDirectory'] = realpath('.')
        data['nni_config'][
            'trialCommand'] = f'python main.py --config {project_parameters_config_filepath}'
        return data, path, project_parameters_config_filepath

    def save_config(self, data: Dict, path: str, config_filepath: str):
        with open(config_filepath, 'w') as stream:
            yaml.dump(data=data, stream=stream)
        nni_config_filepath = path + f'_nni_config.yaml'
        with open(nni_config_filepath, 'w') as stream:
            yaml.dump(data=data['nni_config'], stream=stream)
        return nni_config_filepath

    def run_nni(self, nni_config_filepath: str):
        result = subprocess.run([
            'nnictl', 'create', '--config', nni_config_filepath, '--port',
            str(self.nni_port)
        ])
        return result

    def __call__(self, ) -> Any:
        data, path, project_parameters_config_filepath = self.set_config(
            project_parameters=self.project_parameters)
        nni_config_filepath = self.save_config(
            data=data,
            path=path,
            config_filepath=project_parameters_config_filepath)
        result = self.run_nni(nni_config_filepath=nni_config_filepath)
        return result
