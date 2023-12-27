import os
import sys
import yaml


def read_config():
    """
    read configuration in config.yaml
    :return: a config class
    """
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(proj_dir)
    conf_fp = os.path.join(proj_dir, 'config.yaml')
    with open(conf_fp) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if sys.platform == 'win32':
        nodename = 'Local'
    else:
        nodename = os.uname().nodename

    file_dir = config['filepath'][nodename]
