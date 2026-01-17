import torch
import torch.nn as nn



class DGaedomRsCliModel(nn.Module):
    def __init__(self, net1, indim=4, outdim=2):
        super().__init__()
        self.net1 = net1

        self.sigmoid = nn.Sigmoid()

        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim+1, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=self.outdim)
        self.relu = nn.ReLU()



    def forward(self, x, xcli):

        x1, sfs1, clas2, sfs2, clast, f1, f11, rec = self.net1(x)

        x1 = torch.softmax(x1, dim=1)
        xrscli = torch.cat([x1[:, 1].view(x1.shape[0], -1), xcli], dim=1)
        xrscli = self.linear1(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear2(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear3(xrscli)


        return xrscli, sfs1, clas2, sfs2, clast, f1, f11, rec


class CliModel(nn.Module):
    def __init__(self, indim=4, outdim=2):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=self.outdim)
        self.relu = nn.ReLU()


    def forward(self, xcli):
        xrscli = xcli
        xrscli = self.linear1(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear2(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear3(xrscli)

        return xrscli


