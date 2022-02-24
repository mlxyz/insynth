#  Copyright (c) 2022, Chair of Software Technology
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#  - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  - Neither the name of the University Mannheim nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from urllib import request
from io import BytesIO
from os import path
from zipfile import ZipFile

from tqdm import tqdm


def download_and_unzip(url, extract_to):
    """
    Downloads a zip file from a url and extracts it to a given directory.
    :param url: the url to download the zip file from
    :param extract_to: the directory to extract the zip file to
    :return: 
    """
    if not path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        response = request.urlopen(url)
        bytesio_file = BytesIO()
        with tqdm.wrapattr(bytesio_file, "write",
                           miniters=1, desc=url.split('/')[-1],
                           total=getattr(response, 'length', None)) as file_out:
            for chunk in response:
                file_out.write(chunk)
        zipfile = ZipFile(bytesio_file)
        zipfile.extractall(path=extract_to)
