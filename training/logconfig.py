# -*- coding:utf-8 -*-
import json
import yaml
import logging
import logging.config
import os

default_config = {
    "version": 1, 
    "incremental": False,  
    "disable_existing_loggers": True,  
    "formatters": {  
        "default": {
            "format": "%(name)s %(asctime)s [%(filename)s %(funcName)s()] <%(levelname)s>: %(message)s",
        },
        "brief": {
            "format": "%(name)s [%(funcName)s()] <%(levelname)s>: %(message)s",
        }
    },
    "filters": {},  
    "handlers": {
        "console": {  
            "class": "logging.StreamHandler",
            "formatter": "brief",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "file_detail": {  
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "level": "DEBUG",
            "filename": "detail_logger.log",  
            "encoding": "utf8",
            "maxBytes": 10485760,  
            "backupCount": 10,  
        }
    },
    "loggers": {
        "simple": {
            "level": "DEBUG",
            "handlers": ["console", "file_detail"],
            "propagate": False,  
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file_detail"],
    },
}


def setup_logging(
        default_log_config=None,
        is_yaml=True,
        default_level=logging.INFO,
        env_key='LOG_CFG',
):
    dict_config = None
    logconfig_filename = default_log_config
    env_var_value = os.getenv(env_key, None)

    if env_var_value is not None:
        logconfig_filename = env_var_value

    if default_config is not None:
        dict_config = default_config

    if logconfig_filename is not None and os.path.exists(logconfig_filename):
        with open(logconfig_filename, 'rt', encoding="utf8") as f:
            if is_yaml:
                file_config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                file_config = json.load(f)
        if file_config is not None:
            dict_config = file_config

    if dict_config is not None:
        logging.config.dictConfig(dict_config)
    else:
        logging.basicConfig(level=default_level)


if __name__ == '__main__':
    with open("./logconfig.yaml", "w") as f:
        yaml.dump(default_config, f)

