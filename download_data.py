import urllib.request
import tarfile
import os

datasets_dir = os.path.join(os.path.dirname(__file__), "cs231n", "datasets")
cifar_dir = os.path.join(datasets_dir, "cifar-10-batches-py")

if not os.path.exists(cifar_dir):
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(datasets_dir, "cifar-10-python.tar.gz")
    print("Downloading CIFAR-10...")
    urllib.request.urlretrieve(url, tar_path)
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(datasets_dir)
    os.remove(tar_path)
    print("Done.")
else:
    print("CIFAR-10 already exists.")
