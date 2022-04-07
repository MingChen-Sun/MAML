from torch.utils.data import Dataset
import torch
import numpy as np
from DataGenerate import Task_Generater_Sin,Task_Generater_MINST,Task_Generater_Cora
from dgl.data import CoraGraphDataset as CoraGraphDataset
from torch.utils.data import DataLoader
import torchvision

class MyDataSet(Dataset):
    def __init__(self,task_num,task_generater):
        self.task_generater=task_generater
        self.task_num=task_num
        self.sample_list=[]
        for i in range(self.task_num):
            self.sample_list.append(self.task_generater.generate())
 
    def __getitem__(self, index):
        x= self.sample_list[index][0]
        y= self.sample_list[index][1]
        return x, y
 
    def __len__(self):
        return len(self.sample_list)

if __name__ == "__main__":
    # dataset=MyDataSet(3,Task_Generater(2,4))
    # train_loader = torchvision.datasets.MNIST('./MNIST/',train=True,download=True,
    #                     transform=torchvision.transforms.Compose([
    #                         torchvision.transforms.ToTensor(),
    #                         torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #                         ]))
    # MNIST=Task_Generater_MINST([0,1,2,3,4,5,6,7],2,5,2,train_loader)
    # dataset=MyDataSet(3,MNIST)
    data_source = CoraGraphDataset()
    Cora=Task_Generater_Cora(data_source[0],7,5,10,train_flag=True)
    dataset=MyDataSet(3,Cora)
    print(dataset)
    print(dataset.__len__())
    x,y = dataset.__getitem__(1)
    # print(x,y)
    print(type(x),type(y))
    print(x[0].shape,x[1].shape)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)
    for x,y in dataloader:
        print(type(x),type(y))
        print(len(x),len(y))
        print(x[0].shape,x[1].shape)