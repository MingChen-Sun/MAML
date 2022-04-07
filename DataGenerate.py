from abc import ABCMeta,abstractmethod
import torch
import numpy as np
import random 
import torchvision
from torch.utils.data import WeightedRandomSampler
from dgl.data import CoraGraphDataset as CoraGraphDataset

class Task_Generater(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def sampler(self):
        pass

    def generate(self):
        x_spt,y_spt,x_qry,y_qry=self.sampler()
        x=[x_spt,x_qry]
        y=[y_spt,y_qry]
        return x,y


class Task_Generater_Sin(Task_Generater):
    def __init__(self,sample_num_spt,sample_num_qry):
        super().__init__()
        self.sample_num_spt=sample_num_spt
        self.sample_num_qry=sample_num_qry

    def sampler(self):
        '''
        This part can be rewrote.
        This function means generating the train or test part of a single task.
        '''
        A = np.random.uniform(3, 5.0)
        b = np.random.uniform(0, 0.5*np.pi)

        x_spt = np.random.uniform(-5, 5, self.sample_num_spt)
        x_qry = np.random.uniform(-5, 5, self.sample_num_qry)
        
        y_spt = A * np.sin(x_spt + b)
        y_qry = A * np.sin(x_qry + b)

        x_spt = torch.from_numpy(x_spt).view(-1,1)
        x_qry = torch.from_numpy(x_qry).view(-1,1)
        
        y_spt = torch.from_numpy(y_spt).view(-1,1)
        y_qry = torch.from_numpy(y_qry).view(-1,1)

        x_spt = x_spt.float()
        x_qry = x_qry.float()
        y_qry = y_qry.float()
        y_spt = y_spt.float()
        return torch.FloatTensor(x_spt),torch.FloatTensor(y_spt),torch.FloatTensor(x_spt),torch.FloatTensor(y_spt)


class Task_Generater_MINST(Task_Generater):
    def __init__(self,train_class_list,n_way,k_shot,q_query,data):
        super().__init__()
        self.train_class_list=train_class_list
        self.n_way=n_way
        self.k_shot=k_shot
        self.q_query=q_query
        self.data=data
        self.labels=self.get_labels_dict(train_class_list,self.data)
    
    def get_labels_dict(self,class_list,data):
        labels=[[] for _ in range(len(class_list))]
        label_dict={}
        for index,(x,y) in enumerate(data):
            if y in class_list:
                labels[class_list.index(y)].append(index)
        for i in range(len(class_list)):
            label_dict[str(class_list[i])]=labels[i]
        return label_dict

    def sampler(self):
        if len(self.train_class_list) == self.n_way:
            sampled_classes=self.train_class_list
        else:
            sampled_classes=np.random.choice(self.train_class_list,size=self.n_way)
        
        spt_sampled_index=[]
        qry_sampled_index=[]
        for c in sampled_classes:
            spt_sampled_index.extend(np.random.choice(self.labels[str(c)],size=(self.k_shot)))
            qry_sampled_index.extend(np.random.choice(self.labels[str(c)],size=(self.q_query)))
        
        x_spt=[]
        y_spt=[]
        for i in spt_sampled_index:
            item_x,item_y=self.data[i]
            x_spt.append(torch.unsqueeze(item_x,0))
            y_spt.append(item_y)
        x_spt=torch.cat(x_spt, dim=0)
        y_spt=torch.LongTensor(y_spt)

        x_qry=[]
        y_qry=[]
        for i in qry_sampled_index:
            item_x,item_y=self.data[i]
            x_qry.append(torch.unsqueeze(item_x,0))
            y_qry.append(item_y)
        x_qry=torch.cat(x_qry, dim=0)
        y_qry=torch.LongTensor(y_qry)

        return torch.FloatTensor(x_spt),torch.LongTensor(y_spt),torch.FloatTensor(x_qry),torch.LongTensor(y_qry)

class Task_Generater_Cora(Task_Generater):
    def __init__(self,data_source,n_way,k_shot,q_query,train_flag=True):
        super().__init__()
        self.n_way=n_way
        self.k_shot=k_shot
        self.q_query=q_query
        self.train_flag=train_flag
        self.g=data_source
        self.label_dict,self.class_list=self.decouple_graph(self.train_flag)

    def decouple_graph(self,train_flag=True):
        if train_flag:
            mask = self.g.ndata['train_mask']
        else:
            mask = self.g.ndata['test_mask']

        labels = self.g.ndata['label']
        class_list=list(set(labels.cpu().numpy()))
        label_list=[[] for _ in range(len(class_list))]
        label_dict={}
        for index,y in enumerate(labels):
            if mask[index]:
                label_list[class_list.index(y)].append(index)
        for i in range(len(class_list)):
            label_dict[str(class_list[i])]=label_list[i]
        return label_dict,class_list
        
    def sampler(self):
        if len(self.class_list) == self.n_way:
            sampled_classes=self.class_list
        else:
            sampled_classes=np.random.choice(self.class_list,size=self.n_way)
        
        spt_sampled_index=[]
        qry_sampled_index=[]
        for c in sampled_classes:
            spt_sampled_index.extend(np.random.choice(self.label_dict[str(c)],size=(self.k_shot)))
            qry_sampled_index.extend(np.random.choice(self.label_dict[str(c)],size=(self.q_query)))
        
        x_spt=torch.LongTensor(spt_sampled_index)
        y_spt=torch.LongTensor(spt_sampled_index)
        x_qry=torch.LongTensor(qry_sampled_index)
        y_qry=torch.LongTensor(qry_sampled_index)
        
        return x_spt,y_spt,x_qry,y_qry

if __name__ == "__main__":
    # data=torchvision.datasets.MNIST('./MNIST/',train=True,download=True,
    #                     transform=torchvision.transforms.Compose([
    #                         torchvision.transforms.ToTensor(),
    #                         torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #                         ]))
    # MNIST=Task_Generater_MINST([8,9],2,5,2,data)
    # x,y=MNIST.generate()
    # print(y)
    data_source = CoraGraphDataset()
    Cora=Task_Generater_Cora(data_source[0],2,5,2)
    print(Cora.generate())
    #Cora.decouple_graph()



    pass

