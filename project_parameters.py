# import
import argparse
from yaml import safe_load
from os.path import isfile, isdir, realpath
from typing import Dict, List
import nni

# def


def load_yaml(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        config = safe_load(f)
    assert not (config is None), f'the {filepath} file is empty.'
    return config


# class


class ProjectParameters:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.MetavarTypeHelpFormatter)
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

    def parse_kwargs(self, kwargs: str, kwargs_type: str):
        kwargs_dict = {}
        if kwargs_type in ['str_list', 'num_list']:
            for args in kwargs.split(sep='|'):
                key, value = args.split(sep='=', maxsplit=1)
                if kwargs_type == 'str_list':
                    value = value.split(',')
                elif kwargs_type == 'num_list':
                    value = [eval(v) for v in value.split(',')]
                exec(f'kwargs_dict["{key}"]={value}')
        else:
            for v in kwargs.split(','):
                key, value = v.split(sep='=', maxsplit=1)
                if value in ['None', 'none', 'null', None]:
                    exec(f'kwargs_dict["{key}"]=None')
                elif kwargs_type == 'str':
                    exec(f'kwargs_dict["{key}"]="{value}"')
                elif kwargs_type == 'num':
                    exec(f'kwargs_dict["{key}"]={value}')
                elif kwargs_type == 'bool':
                    exec(f'kwargs_dict["{key}"]=bool({value})')
        return kwargs_dict

    def get_kwargs(self, args: argparse.Namespace):
        kwargs_dict = {}
        for key, value in vars(args).items():
            if value is not None and 'kwargs' in key:
                kwargs_type = key.rsplit('_', 1)[0]
                new_dict = self.parse_kwargs(kwargs=value,
                                             kwargs_type=kwargs_type)
                kwargs_dict.update(new_dict)
        return kwargs_dict

    def set_realpath(self):
        for k, v in self.config.items():
            if isinstance(v, str) and (isfile(v) or isdir(v)):
                self.config[k] = realpath(v)

    def get_keys(self, keys: List = []):
        stack = [['', self.config]]
        while stack:
            root, dic = stack.pop()
            for key, value in dic.items():
                r = root + f'-{key}' if root else key
                if isinstance(value, dict):
                    keys.append(r)
                    stack.append([r, value])
                else:
                    keys.append(r)
        return keys

    def is_valid_kwargs(self, kwargs_dict: Dict, check: bool):
        if check:
            for key in kwargs_dict.keys():
                assert key in self.config_keys, f'please check the keyword argument exists in the configuration.\nkwargs: {key}\nvalid: {self.config_keys}'

    def update(self, kwargs_dict: Dict):
        for key, value in kwargs_dict.items():
            if key in self.config_keys:
                key = key.split('-')
                key = ("['{}']" * len(key)).format(*key)
                try:
                    exec(f'self.config{key}={value}')
                except:
                    exec(f'self.config{key}="{value}"')

    def parse(self):
        args = self.parser.parse_args()
        if isfile(args.config):
            self.config = load_yaml(filepath=args.config)
        else:
            self.config = {}
        self.config['config'] = args.config
        self.config_keys = self.get_keys(keys=[])
        kwargs_dict = self.get_kwargs(args=args)
        self.is_valid_kwargs(kwargs_dict=kwargs_dict, check=args.dont_check)
        self.update(kwargs_dict=kwargs_dict)
        #check nni experiment
        if nni.get_experiment_id() != 'STANDALONE':
            nni_paramter = nni.get_next_parameter()
            self.is_valid_kwargs(kwargs_dict=nni_paramter,
                                 check=args.dont_check)
            self.update(kwargs_dict=nni_paramter)
        if self.config['mode'] == 'train':
            self.config['config_keys'] = self.config_keys
        self.set_realpath()
        return argparse.Namespace(**self.config)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for name, value in vars(project_parameters).items():
        print('{:<20}= {}'.format(name, value))
