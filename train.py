import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import net

def main():
    print("ok")
    n = net(False)
    print("ok")
    print(n(torch.random([10,10,10])))
    print("ok")

if __name__ == '__main__' :
    main()  