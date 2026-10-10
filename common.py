from pathlib import Path
import json
import random

import numpy as np
import seaborn as sns


ROOT=Path(__file__).resolve().parent


def load_config():
    with open(ROOT/'config.json') as f:
        config=json.load(f)
    config.update(config['profiles'][config['run_profile']])
    return config


def task_paths():
    task=Path.cwd().parent
    return task/'input',task/'output',task/'src'


def initialize(seed):
    random.seed(seed)
    np.random.seed(seed)
    sns.set_theme(style='whitegrid',context='notebook',palette='colorblind')
    return np.random.default_rng(seed)


def model_seed(seed,model,offset=0):
    return seed+1009*(['competing_risks','effort_threshold','age_hazards','project_frailty','dynamic_context'].index(model)+1)+offset
