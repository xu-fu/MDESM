
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from mydataset import  Datasetdom4v3
from myModel2D import DGaedomRsCliModel
from AE2D import DisentReconae64
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_curve, auc,  accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from myloss import DGdisentRECaeloss
import os
from loguru import logger
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from visdom import Visdom
from natsort import ns, natsorted
from testdgae224 import test_person_train_test, personclassM


def splitdatatrainval(root):
    ids = os.listdir(root)
    ids = natsorted(ids, alg=ns.PATH)
    xtrain, xval = train_test_split(ids, test_size=0.1)
    return xtrain, xval


def trainssh(root1, trainidlist1, valinlist1, root2, trainidlist2, valinlist2, EPOCH, outtestroot, testpersondict):
    traindata1 = Datasetdom4v3(root1, trainidlist1, root2, trainidlist2, phase='train')
    traindataloader = DataLoader(traindata1, batch_size=64, shuffle=True)#64
    valdata1 = Datasetdom4v3(root1, valinlist1, root2, valinlist2, phase='val')
    valdataloader = DataLoader(valdata1, batch_size=8*2, shuffle=True)
    model5 = DGaedomRsCliModel(net1=DisentReconae64(), indim=5, outdim=2)   
    mydevice = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model5 = model5.to(mydevice)
    myweight = None
    loss = DGdisentRECaeloss(weight1=myweight, weight2=myweight, weight3=myweight)
    optimizer = optim.Adam(params=model5.parameters(), lr=1e-4) 
    loss_list = [] 
    valloss_list = []
    VALLOSSBEST = 99999

    for e in range(EPOCH):

        lossi = 0
        ft = 0
        model5.train()
        
        for batchimgsori, batchimgs, batchlabels, cli, dom in tqdm(traindataloader):
            ft = ft + 1
            batchimgsori = batchimgsori.to(mydevice)
            t2imgs = batchimgs
            t2imgs = t2imgs.to(mydevice)
            cli = cli.to(mydevice)
            batchlabels = batchlabels.to(mydevice)
            dom = dom.to(mydevice)
            tsy = [0]*batchlabels.shape[0] + [1]*dom.shape[0]
            tsy = torch.tensor(tsy).long().to(mydevice)
            batchlabelsn = torch.flip(batchlabels, dims=(0,))
            duiy = batchlabels ^ batchlabelsn
            clas1, sfs1, clas2, sfs2, clast, f1, f11, rec = model5(t2imgs, cli)
            bloss = loss(clas1, batchlabels, clas2, dom, clast, tsy, sfs1[-1], sfs2[-1], f1, f11, duiy, rec, batchimgsori, e) 
            lossi = lossi + bloss.item()
            optimizer.zero_grad()
            bloss.backward()
            optimizer.step()

        
        loss_list.append(lossi / len(traindataloader))  

        print('epoch:{}: train loss:{}'.format(e, loss_list[-1]))

        model5.eval()
        vallossi = 0
        with torch.no_grad():
            for valbatchimgsori, valbatchimgs, valbatchlabels, valcli, valdom in tqdm(valdataloader):
                valbatchimgsori = valbatchimgsori.to(mydevice)
                valt2imgs = valbatchimgs
                valt2imgs = valt2imgs.to(mydevice)
                valcli = valcli.to(mydevice)
                valbatchlabels = valbatchlabels.to(mydevice)
                valdom = valdom.to(mydevice)
                valtsy = [0] * valbatchlabels.shape[0] + [1] * valdom.shape[0]
                valtsy = torch.tensor(valtsy).long().to(mydevice)

                valbatchlabelsn = torch.flip(valbatchlabels, dims=(0,))
                valduiy = valbatchlabels ^ valbatchlabelsn
            
                valclas1, valsfs1, valclas2, valsfs2, valclast, valf1, valf11, valrec = model5(valt2imgs, valcli)
            
                valbloss = loss(valclas1, valbatchlabels, valclas2, valdom, valclast, valtsy,
                                valsfs1[-1], valsfs2[-1], valf1, valf11, valduiy, valrec, valbatchimgsori, e)
                vallossi = vallossi + valbloss.item()

            valloss_list.append(vallossi / len(valdataloader))  
            print('epoch:{}: val loss:{}'.format(e, valloss_list[-1]))


            if valloss_list[-1] < VALLOSSBEST:
                VALLOSSBEST = valloss_list[-1]
                torch.save(model5.state_dict(), 'result/dgaedomweight.pth')

    


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    traintestvalroot = 'train_test_h5_2d-cli4/D1-fuyiyuan'
    traintestvalroot2 = 'train_test_h5_2d-cli4/D2-sanyuan'

    xtrain, xval = splitdatatrainval(os.path.join(traintestvalroot, 'T2'))
    xtrain2, xval2 = splitdatatrainval(os.path.join(traintestvalroot2, 'T2'))
    outtestroot = '/D31-mulyuan'
    testpersondict = personclassM(os.path.join(outtestroot, 'T2'))

    trainssh(root1=traintestvalroot, trainidlist1=xtrain, valinlist1=xval,
             root2=traintestvalroot2, trainidlist2=xtrain2, valinlist2=xval2, EPOCH=200,
             outtestroot=outtestroot, testpersondict=testpersondict)





