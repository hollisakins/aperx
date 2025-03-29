import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'

import numpy as np
from glob import glob
import logging, toml
from dotmap import DotMap

import multiprocessing
total_n_procs = multiprocessing.cpu_count()
n_procs       = int(multiprocessing.cpu_count()/2)

sw_filters = ['f070w','f090w','f115w','f140m','f150w','f162m','f164n','f150w2','f182m','f187n','f200w','f210m','f212n']
lw_filters = ['f250m','f277w','f300m','f322w2','f323n','f335m','f356w','f360m','f405n','f410m','f430m','f444w','f460m','f466n','f470n','f480m']

def setup_logger(name=None, level=logging.DEBUG): # TODO replace with rich logger? 
    if name is None:
        name = __name__
    
    log = logging.getLogger(name)
    log.setLevel(level)

    return log

def parse_config_file(config_file):
    
    # Load the toml config file
    cfg = toml.load(config_file)

    cfg = DotMap(cfg)

    # for key in cfg.environment:
    #     os.environ[key] = cfg.environment[key]
    # del cfg.environment

    cfg.working_dir = os.getcwd()

    return cfg


def get_i2d_filepath(files, path, filtname, prefix, suffix, skip=None):
    pass

def get_sci_filepath():
    pass

def get_err_filepath():
    pass

def get_wht_filepath():
    pass


################################################################################################################################

def Gaussian(x, a, mu, sig):
    return a * np.exp(-(x-mu)**2/(2*sig**2))


def check_files_exist(file_paths):
    """
    Checks if multiple files exist at the given paths.

    Args:
        file_paths: A list of file paths.

    Returns:
        True if all files exist, False otherwise.
    """
    for path in file_paths:
        if not os.path.exists(path):
        #if not os.path.isfile(path): # Use this line to specifically check for files
            return False
    return True

