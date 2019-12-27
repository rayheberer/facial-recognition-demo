"""Download the Yale Faces dataset."""

import os
import requests
import tarfile


url = "https://vismod.media.mit.edu/vismod/classes/mas622-00/datasets/YALE/yalefaces.tar.gz"
download_dir = "data/yale-faces/"
filename = download_dir + url.split("/")[-1]

if not os.path.exists(download_dir):
    os.mkdir(download_dir)

with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)

tar = tarfile.open(filename, "r:gz")
tar.extractall(download_dir)
