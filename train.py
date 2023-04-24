import torch

from model.net import Net

def main():
    n = Net(False)
    #n.to('cuda')

    img = torch.rand(1, 3, 640, 512)
    img = n.forward(img)
    for i in img:
        print(i.shape)

if __name__ == '__main__' :
    main()  