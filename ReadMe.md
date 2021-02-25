# Character Recognizer with PyTorch
> This project aims to show how to use PyTorch to recognize handwritten characters.

## Before we begin, we must make sure to setup!
1. We first clone this repo
```bash
$ git clone https://github.com/mnist_recognizer_pytorch
$ cd mnist_recognizer_pytorch
```

2. Create a new environment with conda and install requirements
```bash
$ conda create --name mnist_recognizer_pytorch python=3.7
$ conda activate mnist_recognizer_pytorch
$ pip install -r requirements.txt
```


## Using PyTorch to Recognize Characters
We can run an experiment, for now, using two public datasets:
- **MNIST**: MNIST stands for Mini-NIST, where NIST is the National Institute of Standards and Technology, which compiled a dataset of handwritten digits and letters in the 1980s. MNIST is Mini because it only included digits.
- **EMNIST**: EMNIST is a repackaging of the original dataset, which also includes letters, but presented in the popularized MNIST format. You can see a publication about it [here](https://www.paperswithcode.com/paper/emnist-an-extension-of-mnist-to-handwritten).

After installing the libraries, use python to run `run_experiment.py`
```bash
$ export PYTHONPATH=. # Defines root as python workdir
$ python training/run_experiment.py --max_epochs=5 --gpus=0 --num_workers=20 --model_class=MLP --data_class=MNIST
```
Where:
- **--max_epochs**: Controls the number of complete passes through the training dataset.
- **--gpus**: ID of GPU used to train the model. 0 stands for `CPU` and 1 for `GPU` or -1 to `use all GPUs available`.
- **--num_workers**: How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
- **--model_class**: Class for modeling available in `model_core.models`. Contains the Neural Network Architecture.
- **--data_class**: Class for downloading and preparing data available in `model_core.data`. Contains all steps to download and preprocess data. It can be `MNIST` or `EMNIST`.

---
P.S.: Project based on labs from [Spring 2021 Full Stack Deep Learning Course](https://www.fullstackdeeplearning.com/)
