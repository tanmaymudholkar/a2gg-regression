import argparse
import a2gg_training_env

import torch
print("torch version: {v}".format(v=torch.__version__))
from torch import nn
from torch.utils.data import DataLoader
import torchvision
print("torchvision version: {v}".format(v=torchvision.__version__))
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib
print("matplotlib version: {v}".format(v=matplotlib.__version__))
import matplotlib.pyplot as plt

inputArgumentsParser = argparse.ArgumentParser(description='Basic template.')
inputArgumentsParser.add_argument('--test', default="testarg", help="Test argument.", type=str)
inputArguments = inputArgumentsParser.parse_args()

arg_test = inputArguments.test
print(f"Argument \"test\": {arg_test}")

print("cuda is_available: {b}".format(b=torch.cuda.is_available()))

a2gg_training_env.execute_in_env(commandToRun="echo template")

print("All done!")
