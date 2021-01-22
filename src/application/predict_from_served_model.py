import base64

import pandas as pd
import requests

from src.settings.settings import logging, paths
from src.tools.utils import read_image

# from PIL import TiffImagePlugin

# TiffImagePlugin.DEBUG = False


def call_pred(imgs_paths, host, path, port=None):
    """Calls the model served to get predictions.

    Parameters
    ----------
    imgs_paths : list
        List of paths of images to make a prediction.
    host : str
        Address of the server where the model is served.
    port : int
        Port of the server where the model is served.

    Returns
    -------
    response
        Response for the HTTP call.
    """

    data = pd.DataFrame(
        data=[base64.encodebytes(read_image(x)) for x in imgs_paths],
        columns=["image"],
    ).to_json(orient="split")

    logging.debug("Data : {}".format(data))

    if port is not None:
        url = "{host}:{port}/{path}".format(host=host, port=port, path=path)
    else:
        url = "{host}/{path}".format(host=host, path=path)

    headers = {"Content-Type": "application/json; format=pandas-split"}
    key = "wJnVBnhcAPjY6G3wbEt14wvo1QKWMc9Y"
    headers["Authorization"] = f"Bearer {key}"

    response = requests.post(
        url=url,
        data=data,
        headers=headers,
    )
    return response


response = call_pred(
    imgs_paths=paths["images"],
    host=("http://40.89.187.133:80/api/v1/service/test-food-service/"),
    path="score",
    port=None,
)
logging.info("Response {}".format(response.content))
