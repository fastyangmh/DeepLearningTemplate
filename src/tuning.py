#import
from ray import tune
import torch
import ray
from copy import deepcopy

#class
class BaseTuning:   #TODO: finish BaseTuning.
    def __init__(self, hyperparameter_space_config,
                 gpu_resources_per_trial) -> None:
        self.hyperparameter_space = self.parse_hyperparameter_space(
            hyperparameter_space_config=hyperparameter_space_config)
        self.gpu_resources_per_trial = gpu_resources_per_trial if torch.cuda.is_available(
        ) else 0

    def parse_hyperparameter_space(self, hyperparameter_space_config):
        hyperparameter_space = {}
        for key, value in hyperparameter_space_config.items():
            for typ, arguments in value.items():
                hyperparameter_space_arguments = []
                for a, b in arguments.items():
                    if type(b) is str:
                        arg = '{}="{}"'.format(a, b)
                    else:
                        arg = '{}={}'.format(a, b)
                    hyperparameter_space_arguments.append(arg)
                hyperparameter_space_arguments = ','.join(
                    hyperparameter_space_arguments)
                arguments = hyperparameter_space_arguments
                arguments = ','.join(arguments)
                hyperparameter_space[key] = eval('tune.{}({})'.format(
                    typ, arguments))
        return hyperparameter_space

    def get_tuning_parameters(self, hyperparameter_space, project_parameters):
        for k, v in hyperparameter_space.items():
            if type(v) == str:
                exec('project_parameters.{}="{}"'.format(k, v))
            else:
                exec('project_parameters.{}={}'.format(k, v))
        project_parameters.num_workers = project_parameters.cpu_resources_per_trial
        return project_parameters

    def parse_training_result(self, result):
        train_loss = result['train']['test_loss']
        val_loss = result['val']['test_loss']
        return train_loss, val_loss

    def parse_tuning_result(self, result, metric, mode, project_parameters):
        best_trial = result.get_best_trial(metric=metric, mode=mode)
        if not project_parameters.tuning_test:
            project_parameters = self.get_tuning_parameters(
                hyperparameter_space=best_trial.config,
                project_parameters=deepcopy(project_parameters))
            result = train(project_parameters=project_parameters)
            result['tuning'] = tuning_result
        else:
            result = {'tuning': tuning_result}
        print('-' * 80)
        print('best trial name: {}'.format(best_trial))
        print('best trial result: {}'.format(
            best_trial.last_result['sum_of_train_and_val_loss']))
        best_config = [
            '{}: {}'.format(a, b) for a, b in best_trial.config.items()
        ]
        best_config = '\n'.join(best_config)
        print('best trial config:\n\n{}'.format(best_config))
        print('num_workers: {}'.format(
            project_parameters.cpu_resources_per_trial))
