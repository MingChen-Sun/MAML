import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from Get_Args import get_args
from Model import Net,CNN,CNN_Casual,GCN,GCN_Filter
from torch.utils.data import DataLoader
from Dataset import MyDataSet
from DataGenerate import Task_Generater_Sin,Task_Generater_MINST,Task_Generater_Cora
import matplotlib.pyplot as plt
import time
import torchvision
from dgl.data import CoraGraphDataset as CoraGraphDataset

class MAML(nn.Module):
    def __init__(self, model,args):
        '''
        model input base model type: nn.Moudle
        i_lf  inner loss function type: function
        o_lf  outer loss function type: function
        f_lf  fine-tuning losss funtion type: function
        i_opt inner optimizer type:
        o_opt outer optimizer type:
        f_opt fine-tuning optimizer type:
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
        
    def inject_graph(self,g):
        self.g=g.to(self.args.device)

    def get_model(self):
        return copy.deepcopy(self.model)

    def forward(self, dataloader, i_os=1):
        '''
        i_os inner optimize step per task defult 1 type: int
        '''
        ########
        features = self.g.ndata['feat'].to(self.args.device)
        labels = self.g.ndata['label'].to(self.args.device)
        ########
        everage_loss=0
        for index,(batch_x, batch_y) in enumerate(dataloader):
            self.train()
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
                    y_spt_pre=self.model(features)
                    i_spt_loss=self.i_lf(y_spt_pre[batch_x[i][0]],labels[batch_y[i][0]])#loss function input target
                    self.i_opt.zero_grad()
                    i_spt_loss.backward()
                    self.i_opt.step()
                parameters_list.append(copy.deepcopy(self.model.get_parameters()))
            o_loss=0
            for i in range(self.args.batch_size):
                self.model.load_parameters(parameters_list[i])
                y_qry_pre=self.model(features)
                i_qry_loss=self.i_lf(y_qry_pre[batch_x[i][1]],labels[batch_y[i][1]])
                o_loss=o_loss+(i_qry_loss)

            theta=copy.deepcopy(theta_)
            self.model.load_parameters(theta)
            o_loss=o_loss/len(batch_x)
            self.o_opt.zero_grad()
            o_loss.backward()
            self.o_opt.step()
            everage_loss=everage_loss+o_loss
        everage_loss=everage_loss/len(dataloader)
        return everage_loss

    def evaluation(self,fine_tuning_step,task):
        ########
        features = self.g.ndata['feat'].to(self.args.device)
        labels = self.g.ndata['label'].to(self.args.device)
        ########
        x,y=task
        device=self.args.device
        x_spt,x_qry=x
        y_spt,y_qry=y

        model_copy=self.get_model()
        f_lf=nn.CrossEntropyLoss()
        f_opt=torch.optim.Adam(model_copy.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        for _ in range(fine_tuning_step):
            model_copy.train()
            y_pre=model_copy(features.to(device))

            loss=f_lf(y_pre[x_spt],labels[y_spt].to(device))
            f_opt.zero_grad()
            loss.backward()
            f_opt.step()
            

        model_copy.eval()
        output=model_copy(features.to(device))
        prediction = torch.argmax(output[x_qry], 1)
        correct = (prediction == labels[y_qry].to(device)).sum().float()/labels[y_qry].shape[0]
        return correct
        
        

if __name__ == "__main__":
    
    args=get_args()
    data_source = CoraGraphDataset()
    model = GCN_Filter(data_source[0].to(args.device),args.input_size, args.hidden_size, args.output_size, F.relu, 0.3)
    # model = GCN(data_source[0].to(args.device),args.input_size, args.hidden_size, args.output_size, F.relu, 0.3)
    maml=MAML(model,args)
    maml.inject_graph(data_source[0].to(args.device))
    Cora_train=Task_Generater_Cora(data_source[0].to(args.device),7,5,10,train_flag=True)
    Cora_test=Task_Generater_Cora(data_source[0].to(args.device),7,50,100,train_flag=False)
    loss_t=[]
    for e in range(1000):
        dataset = MyDataSet(args.task_num,Cora_train)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        loss=maml(dataloader)
        loss_t.append(loss.cpu().detach().numpy())
        if e%25==0:
            print(loss)
            ##########
            acc=maml.evaluation(20,Cora_test.generate())
            print("Test ACC value : ",acc.cpu().numpy())
            ##########
    
    plt.figure()
    plt.plot(loss_t)
    plt.savefig('./pic.png')

    


    #Test
    # plt.figure(figsize=(20,10))
    # for i in range(10):
    #     x,y=Task_Generater(args.sample_num_spt,args.sample_num_qry).sampler(100)
    #     y_pre=maml.get_model()(x)
    #     plt.subplot(2,5,i+1)
    #     plt.scatter(x,y)
    #     plt.scatter(x,y_pre.detach().numpy())
    # plt.savefig('./exa.png')


