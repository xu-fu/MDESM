
import torch
from torch.utils.data import DataLoader
from mydataset import  Dataset4v2
from myModel2D import DGBaselineRsCliModel, DGaedomRsCliModel, DGaedomRsCliModelnodisent, CliModel
from myAE2D import DisentRecon
from AE2D import DisentReconae64, DisentReconae224, Bsaelineae, Reconae64
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_curve, auc,  accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from myloss import SSIM
from natsort import ns, natsorted
from tqdm import tqdm
import random


def splitdatatraintestval(root):
    
    ids = os.listdir(root)
    ids = natsorted(ids, alg=ns.PATH)
    xtrain, xtestval = train_test_split(ids, test_size=0.15)
    xtest, xval = train_test_split(xtestval, test_size=0.5)
    return xtrain, xval, xtest
def splitdatatrainval(root):
    
    ids = os.listdir(root)
    ids = natsorted(ids, alg=ns.PATH)
    xtrain, xval = train_test_split(ids, test_size=0.1)
    return xtrain, xval



def test_person_rs_dis_rec(root, persondict, weight, flg='max'):
    # 加载模型
    model5 = DisentReconae64()

    # 加载权重
    mydevice = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model5.load_state_dict(torch.load(weight, map_location=mydevice))
    model5 = model5.to(mydevice)
    model5.eval()
    
    testloss_list = []  
    testlossi = 0
    test_y_heat = [] 
    test_y_true = [] 
    test_y_per = [] 

    allp = natsorted(persondict.keys(), alg=ns.PATH)
    for idi in allp:
        testidilist = persondict[idi]
        bs = len(testidilist)

        testdata = Dataset4v2(root, testidilist, phase='test')
        testdataloader = DataLoader(testdata, batch_size=bs, shuffle=False)


        with torch.no_grad():
            for _, testbatchimgsori, testbatchimgs, testbatchlabels, testcli in tqdm(testdataloader):
                testt2imgs = testbatchimgs
                
                testt2imgs = testt2imgs.to(mydevice)
                testbatchlabels = testbatchlabels.to(mydevice)
                testcli = testcli.to(mydevice)
                testbatchimgsori = testbatchimgsori.to(mydevice)
                
                testclas1, testsfs1, testclas2, testsfs2, testclast, testf1, testf11, testrec = model5(testt2imgs)

                

                testbatchlabelsy = testbatchlabels.data.cpu().numpy() 


                if flg == 'max':
                    testpred_y_softmax = torch.softmax(testclas1, dim=1).detach().cpu().numpy()

                    testpred_y_softmaxaddperson = testpred_y_softmax[int(np.argmax(testpred_y_softmax)/2)]
                    testpredictedperson = np.argmax(testpred_y_softmaxaddperson)  


                
                test_y_heat.extend([testpredictedperson]) 
                test_y_true.extend([testbatchlabelsy[0]]) 
                test_y_per.extend(list(testpred_y_softmaxaddperson)) 
                testloss_list.append(testlossi / len(testdataloader)) 

    testacc = accuracy_score(test_y_true, test_y_heat)
    test_y_truearr = np.array(test_y_true)
    test_y_perarr = np.array(test_y_per).reshape([-1, 2])
    test_y_heatarr = np.array(test_y_heat)
    test_y_true_save = test_y_truearr
    test_y_true_save = test_y_true_save.reshape([-1, 1])
    test_y_per_save = test_y_perarr
    test_y_heat_seave = test_y_heatarr.reshape([-1, 1])
    test_save = np.concatenate([test_y_true_save, test_y_heat_seave, test_y_per_save], axis=1)
    pd.DataFrame(test_save).to_csv('result/predictresultrs_rs_nomult.csv')
    

def personclassM(inroot):
    idslist = os.listdir(inroot)
    idl = []
    for idi in idslist:
        idl.append('{}-{}'.format(idi.split('-')[0], idi.split('-')[1]))
    idl = list(set(idl))
    iddict = {}
    for idli in idl:
        iddict[idli] = []
    for idii in idslist:
        id1 = '{}-{}'.format(idii.split('-')[0], idii.split('-')[1])
        iddict[id1].append(idii)
    return iddict


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    outtestroot =  'D31-mulyuan' 
    xtest = os.listdir(os.path.join(outtestroot, 'T2'))
    testpersondict = personclassM(os.path.join(outtestroot, 'T2'))
    test_person_rs_dis_rec(root=outtestroot, persondict=testpersondict,
                           weight='dgaedomweight.pth', flg='max')

