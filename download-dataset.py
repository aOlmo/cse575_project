from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from urllib.request import urlopen  # python 3
except ImportError:
    from urllib2 import urlopen  # python 2
import os
import tarfile
import tempfile
import shutil

dataset = 'images'  # Stanford Dog dataset is called images.tar
url = "http://vision.stanford.edu/aditya86/ImageNetDogs/%s.tar" % dataset
with tempfile.TemporaryFile() as tmp:
    print("downloading", url)
    shutil.copyfileobj(urlopen(url), tmp)
    print("extracting")
    tmp.seek(0)
    tar = tarfile.open(fileobj=tmp)
    tar.extractall()
    tar.close()
    # print("renaming folder to 'data'")
    # os.rename('Images/', 'data/')
    print("moving pugs to data folder")
    files = os.listdir('Images/n02110958-pug/')
    for f in files:
        shutil.move('Images/n02110958-pug/' + f, 'data/')
    # print("removing extra dog images")
    # os.remove('Images/')
    print("done")
