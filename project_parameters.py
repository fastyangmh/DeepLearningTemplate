# import
import argparse
from os.path import abspath, isdir, isfile
from typing import Dict

import nni
import numpy as np
from yaml import safe_load

# def


def load_yaml(filepath: str):
    with open(file=filepath, mode='r', encoding='utf-8') as stream:
        config = safe_load(stream=stream)
    assert not (config is None), f'the {filepath} file is empty.'
    config['config'] = filepath
    return config


# class


class ProjectParameters:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.MetavarTypeHelpFormatter)
        self.add_argument()

    def add_argument(self):
        self.parser.add_argument(
            '--config',
            type=str,
            required=True,
            help=
            'the project configuration path. if given None, it will not be loaded (it needs to be used with dont_check).'
        )
        self.parser.add_argument(
            '--dont_check',
            action='store_false',
            help=
            'whether to check kwargs, if given, kwargs will not be checked.')
        self.parser.add_argument(
            '--str_kwargs',
            type=str,
            help='the keyword whose value type is a string.')
        self.parser.add_argument(
            '--num_kwargs',
            type=str,
            help='the keyword whose value type is a number.')
        self.parser.add_argument(
            '--bool_kwargs',
            type=str,
            help='the keyword whose value type is a boolean.')
        self.parser.add_argument(
            '--str_list_kwargs',
            type=str,
            help=
            'the keyword whose value type is a list of strings. please use the "|" symbol as separator, if you have multiple keys to modify.'
        )
        self.parser.add_argument(
            '--num_list_kwargs',
            type=str,
            help=
            'the keyword whose value type is a list of numbers. please use the "|" symbol as separator, if you have multiple keys to modify.'
        )
        self.parser.add_argument(
            '--bool_list_kwargs',
            type=str,
            help=
            'the keyword whose value type is a list of booleans. please use the "|" symbol as separator, if you have multiple keys to modify.'
        )

    def parse(self):
        args = self.parser.parse_args()

        if isfile(args.config):
            self.config = load_yaml(filepath=args.config)
        else:
            self.config = {}

        self.config_keys = self.get_config_keys(config=self.config)
        kwargs_group = self.parse_args(args=args)
        self.is_valid_kwargs(kwargs_group=kwargs_group, check=args.dont_check)
        self.update_config(kwargs_group=kwargs_group)

        #check nni experiment
        self.get_nni_parameter(args)

        if 'mode' in self.config and self.config['mode'] == 'train':
            self.config['config_keys'] = self.config_keys

        self.set_path_to_abs()

        #parse classes if classes is filepath
        self.parse_classes()

        return argparse.Namespace(**self.config)

    def parse_classes(self):
        if 'classes' in self.config and isinstance(self.config['classes'],
                                                   str) and isfile(
                                                       self.config['classes']):
            self.config['classes'] = np.loadtxt(fname=self.config['classes'],
                                                dtype=str).tolist()

    def get_nni_parameter(self, args):
        if nni.get_experiment_id() != 'STANDALONE':
            nni_paramter = nni.get_next_parameter()
            self.is_valid_kwargs(kwargs_dict=nni_paramter,
                                 check=args.dont_check)
            self.update_config(kwargs_dict=nni_paramter)

    @staticmethod
    def get_config_keys(config):
        stack = [['', config]]
        keys = []
        while len(stack):
            root, dic = stack.pop()
            for key, value in dic.items():
                key = root + f'-{key}' if root else key
                keys.append(key)
                if isinstance(value, dict):
                    stack.append([key, value])
        return keys

    def parse_args(self, args: argparse.Namespace):
        kwargs_group = {}
        for key, value in vars(args).items():
            if value is not None and 'kwargs' in key:
                kwargs_type = key.rsplit('_', 1)[0]
                kwargs_group.update(
                    self.parse_kwargs(kwargs=value, kwargs_type=kwargs_type))
        return kwargs_group

    @staticmethod
    def parse_kwargs(kwargs: str, kwargs_type: str):
        kwargs_group = {}
        if 'list' in kwargs_type:
            for args in kwargs.split(sep='|'):
                key, value = args.split(sep='=', maxsplit=1)
                if kwargs_type == 'str_list':
                    value = value.split(',')
                elif kwargs_type == 'num_list':
                    value = [eval(v) for v in value.split(',')]
                elif kwargs_type == 'bool_list':
                    value = [bool(v) for v in value.split(',')]
                exec(f'kwargs_group["{key}"]={value}')
        else:
            for args in kwargs.split(','):
                key, value = args.split(sep='=', maxsplit=1)
                if value in ['None', 'none', 'null', None]:
                    exec(f'kwargs_group["{key}"]=None')
                elif kwargs_type == 'str':
                    exec(f'kwargs_group["{key}"]="{value}"')
                elif kwargs_type == 'num':
                    exec(f'kwargs_group["{key}"]={value}')
                elif kwargs_type == 'bool':
                    exec(f'kwargs_group["{key}"]=bool({value})')
        return kwargs_group

    def is_valid_kwargs(self, kwargs_group: Dict, check: bool):
        if check:
            for key in kwargs_group.keys():
                assert key in self.config_keys, f'please check the keyword argument exists in the configuration.\nkwargs: {key}\nvalid: {self.config_keys}'

    def update_config(self, kwargs_group: Dict):
        for key, value in kwargs_group.items():
            key = key.split('-')
            key = ("['{}']" * len(key)).format(*key)
            try:
                exec(f'self.config{key}={value}')
            except:
                exec(f'self.config{key}="{value}"')

    def set_path_to_abs(self):
        for k, v in self.config.items():
            if isinstance(v, str) and (isfile(v) or isdir(v)):
                self.config[k] = abspath(v)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for name, value in vars(project_parameters).items():
        print('{:<20}= {}'.format(name, value))
