import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from Get_Args import get_args
from Model import Net,CNN,CNN_Casual
from torch.utils.data import DataLoader
from Dataset import MyDataSet
from DataGenerate import Task_Generater,Task_Generater_MINST
import matplotlib.pyplot as plt
import time
import torchvision

class MAML(nn.Module):
    def __init__(self, model,args):
        '''
        model input base model type: nn.Moudle
        i_lf inner loss function type: function
        o_lf outer loss function type: function
        i_opt inner optimizer type:
        o_opt outer optimizer type:
        '''
        super(MAML, self).__init__()
        self.args=args
        self.model=copy.deepcopy(model)
        self.model.to(self.args.device)
        '''
        The following part can be replaced by other optimizers or loss funcitons.
        '''
        self.i_lf=nn.CrossEntropyLoss()#nn.MSELoss()
        self.o_lf=nn.CrossEntropyLoss()#nn.MSELoss()
        self.i_opt=torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.o_opt=torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def get_model(self):
        return copy.deepcopy(self.model)

    def forward(self, dataloader, i_os=1):
        '''
        i_os inner optimize step per task defult 1 type: int
        '''
        self.train()
        everage_loss=0
        for index,(batch_x, batch_y) in enumerate(dataloader):
            theta_=copy.deepcopy(self.model.get_parameters())
            parameters_list=[]
            execute_x=[[batch_x[0][i].to(self.args.device),batch_x[1][i].to(self.args.device)] for i in range(self.args.batch_size)]
            execute_y=[[batch_y[0][i].to(self.args.device),batch_y[1][i].to(self.args.device)] for i in range(self.args.batch_size)]
            batch_x=execute_x
            batch_y=execute_y
            for i in range(self.args.batch_size):
                theta=copy.deepcopy(theta_)
                self.model.load_parameters(theta)
                for _ in range(i_os):
                    y_spt_pre=self.model(batch_x[i][0])
                    i_spt_loss=self.i_lf(y_spt_pre,batch_y[i][0])#loss function input target
                    #### constraint
                    mask_matrix=maml.get_model().get_mask_matrix()
                    loss_c=(mask_matrix**2).sum() / 2
                    i_spt_loss=i_spt_loss+args.lc*loss_c
                    ####
                    self.i_opt.zero_grad()
                    i_spt_loss.backward()
                    self.i_opt.step()
                parameters_list.append(copy.deepcopy(self.model.get_parameters()))
            o_loss=0
            for i in range(self.args.batch_size):
                self.model.load_parameters(parameters_list[i])
                y_qry_pre=self.model(batch_x[i][1])
                i_qry_loss=self.i_lf(y_qry_pre,batch_y[i][1])
                o_loss=o_loss+(i_qry_loss)
            
            theta=copy.deepcopy(theta_)
            self.model.load_parameters(theta)

            o_loss=o_loss/len(batch_x)
            #### constraint
            mask_matrix=maml.get_model().get_mask_matrix()
            loss_c=(mask_matrix**2).sum() / 2
            o_loss=o_loss+args.lc*loss_c
            ####
            self.o_opt.zero_grad()
            o_loss.backward()
            self.o_opt.step()
            everage_loss=everage_loss+o_loss
        everage_loss=everage_loss/len(dataloader)
        return everage_loss

    
        
        

if __name__ == "__main__":
    args=get_args()
    #model = CNN()#Net(args.input_size, args.hidden_size, args.output_size, F.relu)
    model = CNN_Casual()

    maml=MAML(model,args)
    #data=torchvision.datasets.MNIST('./MNIST/',train=True,download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    data=torchvision.datasets.MNIST('./MNIST/',train=True,download=True,
                        transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    MNIST=Task_Generater_MINST([0,1,2,3,4,5,6,7],3,2,5,data)

    loss_t=[]
    for e in range(1000):
        dataset = MyDataSet(args.task_num,MNIST)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        loss=maml(dataloader)
        loss_t.append(loss.cpu().detach().numpy())
        if e%10==0:
            print(loss)
    
    plt.figure()
    plt.plot(loss_t)
    plt.savefig('./pic.png')
    
    '''
    '''
    print(maml.get_model().get_mask_matrix().cpu().numpy())
    #######
    plt.figure()
    mask_matrix=maml.get_model().get_mask_matrix().cpu().numpy()
    plt.imshow(mask_matrix)#,cmap='Greys_r')
    plt.savefig('./mask_matrix.png')
    #######
    
    #Test
    plt.figure(figsize=(100,50))
    for i in range(21):
        plt.subplot(3,7,i+1)
        plt.imshow(data[i][0][0])#,cmap='Greys_r')
    plt.savefig('./pic_data_o.png')   

    plt.figure(figsize=(100,50))
    for i in range(21):
        plt.subplot(3,7,i+1)
        mask_matrix=maml.get_model().get_mask_matrix().cpu()
        x_mask=data[i][0][0]*mask_matrix
        plt.imshow(x_mask)#,cmap='Greys_r')
    plt.savefig('./pic_data.png')


