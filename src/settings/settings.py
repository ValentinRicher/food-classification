import logging
from pathlib import Path

import yaml
from PIL import TiffImagePlugin
from pyhocon import ConfigFactory

LOGGING_FORMAT = "[%(asctime)s][%(levelname)s][%(module)s] %(message)s"
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_LEVEL = logging.DEBUG
logging.basicConfig(
    format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT, level=LOGGING_LEVEL
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

TiffImagePlugin.DEBUG = False


ENV = "qbox"  # the name of the environment matches the name of the folder in settings


paths = ConfigFactory.parse_file(Path(__file__).parent.joinpath(str(ENV), "paths.conf"))

with open(
    Path(__file__).parent.joinpath("common", "models", "mobilenetv2_params.yaml"),
    "r",
) as f:
    mobilenetv2_params = yaml.load(f, Loader=yaml.FullLoader)

with open(
    Path(__file__).parent.joinpath("common", "models", "xception_params.yaml"),
    "r",
) as f:
    xception_params = yaml.load(f, Loader=yaml.FullLoader)


if Path.exists(Path(__file__).parent.joinpath("common", "azure_config.conf")):
    azure_config = ConfigFactory.parse_file(
        Path(__file__).parent.joinpath("common", "azure_config.conf")
    )
