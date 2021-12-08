import os
import urllib
from io import BytesIO
from os import path
from zipfile import ZipFile

from tqdm import tqdm


def download_and_unzip(url, extract_to):
    if not path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        response = getattr(urllib, 'request', urllib).urlopen(url)
        bytesio_file = BytesIO()
        with tqdm.wrapattr(bytesio_file, "write",
                           miniters=1, desc=url.split('/')[-1],
                           total=getattr(response, 'length', None)) as file_out:
            for chunk in response:
                file_out.write(chunk)
        zipfile = ZipFile(bytesio_file)
        zipfile.extractall(path=extract_to)
