import yaml


def read_yaml_file(file_dir):
    with open(file_dir, 'r') as f:
        config = yaml.safe_load(f)
    return config


