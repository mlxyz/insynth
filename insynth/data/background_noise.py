from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


def download_and_unzip(url, extract_to='./data/esc-50/'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

