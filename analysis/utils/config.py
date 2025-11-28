import copy
import os.path as op
import yaml

CONFIG_DIR_REL = "../../../config"
CONFIG_DIR = op.realpath(op.join(__file__ , CONFIG_DIR_REL))

DEFAULT_MODEL_CONFIG_PATH = op.join(CONFIG_DIR, "model/main/change-point.yaml")
DEFAULT_TASK_CONFIG_PATH = op.join(CONFIG_DIR, "task/change-point-env.yaml")

def config_path_with_name(fname, subdir=None):
    if subdir:
        return op.join(CONFIG_DIR, subdir, fname)
    else:
        return op.join(CONFIG_DIR, fname)

def load_config_at_path(config_path, add_id=True):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    if add_id and config.get('id', None) is None:
        config['id'] = op.splitext(op.basename(config_path))[0]
    return config

def load_config_with_name(fname, subdir=None, add_id=True):
    return load_config_at_path(config_path_with_name(fname, subdir=subdir,),
        add_id=add_id)

def setdefaults(config, defaults):
    for k, v in defaults.items():
        config.setdefault(k, v)

def load_config_at_path_with_default(config_path=None, default_config={},
    do_setdefaults=False):
    if config_path is None:
        config = default_config
    else:
        config = load_config_at_path(config_path)
        if do_setdefaults:
            setdefaults(config, default_config)
    return config

def load_default_model_config():
    return load_config_at_path(DEFAULT_MODEL_CONFIG_PATH)

def load_default_task_config():
    return load_config_at_path(DEFAULT_TASK_CONFIG_PATH)

def load_config_at_path_with_default_model_config(config_path=None):
    return load_config_at_path_with_default(config_path=config_path,
        default_config=load_default_model_config())

def load_config_at_path_with_default_task_config(config_path=None):
    return load_config_at_path_with_default(config_path=config_path,
        default_config=load_default_task_config())

def create_joint_config(task_config, model_config=None, replace_value="true"):
    config = {
        "task": task_config.copy(),
        "model": (model_config.copy() if model_config is not None else None),
    }
    config["id"] = (f"task-{task_config['id']}-model-{model_config['id']}"
                    if model_config is not None else f"task-{task_config['id']}-model-none")
    if model_config is not None and "inference" in model_config:
        config["model"]["inference"] = _recursive_replace(
                config["model"]["inference"], task_config, replace_value=replace_value)
    return config

def _recursive_replace(original_value, ref_value, replace_value="true"):
    if (original_value == replace_value):
        return copy.deepcopy(ref_value)
    elif not isinstance(original_value, dict):
        return original_value
    else:
        new_dict = {}
        for k, original_v in original_value.items():
            if k in ref_value:
                new_dict[k] = _recursive_replace(original_v, ref_value[k])
            else:
                new_dict[k] = copy.deepcopy(original_v)
        return new_dict
