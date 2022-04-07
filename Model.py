import torch
import torch.nn as nn
import torch.nn.functional as F
from Get_Args import get_args
from scipy.sparse.linalg import eigs
from scipy.sparse import diags
import scipy as sp
import numpy as np

class Net_base(nn.Module):
    def __init__(self):
        super(Net_base, self).__init__()
        pass

    def get_parameters(self):
        parameters=[]
        for name, param in self.named_parameters():
            parameters.append([name,param.data.clone()])
        return parameters

    def load_parameters(self,fast_parameters):
        for index,param in enumerate(self.parameters()):
            param.data=fast_parameters[index][1]

class Net(Net_base):
    def __init__(self, input_size, hidden_size, output_size,activation,dropout=None):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.activation=activation
        if dropout !=None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout=dropout
        if len(hidden_size) < 1:
            self.layers.append(nn.Linear(input_size, output_size))
        else:
            self.layers.append(nn.Linear(input_size, hidden_size[0]))
            for i in range(len(hidden_size)-1):
                self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.layers.append(nn.Linear(hidden_size[-1], output_size))
    
    def forward(self, x):
        h=x
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != len(self.layers) - 1:
                if self.activation!=None:
                    h = self.activation(h)
                if self.dropout!=None:
                    h = self.dropout(h)
        return h

from dgl.nn.pytorch import GraphConv

class GCN(Net_base):
    def __init__(self,g,input_size,hidden_size,output_size,activation,dropout=None):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        if len(hidden_size) < 1:
            self.layers.append(GraphConv(input_size, output_size, activation=activation))
        else:
            self.layers.append(GraphConv(input_size, hidden_size[0], activation=activation))
            for i in range(len(hidden_size)-1):
                self.layers.append(GraphConv(hidden_size[i], hidden_size[i+1], activation=activation))
            self.layers.append(GraphConv(hidden_size[-1], output_size, activation=activation))
        if dropout !=None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout=dropout

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

class GCN_Filter(Net_base):
    def __init__(self,g,input_size,hidden_size,output_size,activation,dropout=None):
        super(GCN_Filter, self).__init__()
        self.g = g
        self.values,self.vector=self.execute_structure_info(self.g)
        dim=self.values.shape[1]
        self.layers = nn.ModuleList()
        self.activation=activation
        if dropout !=None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout=dropout
        self.layers_f=nn.Linear(dim,dim)
        if len(hidden_size) < 1:
            self.layers.append(nn.Linear(input_size, output_size)) 
        else:
            self.layers.append(nn.Linear(input_size, hidden_size[0]))       
            for i in range(len(hidden_size)-1):
                self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))              
            self.layers.append(nn.Linear(hidden_size[-1], output_size))
    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    def execute_structure_info(self,g):
        print("*****************************")
        # adj=g.adj().to_dense()
        # D=torch.diag(adj.sum(0))
        # D=torch.pow(D,-1/2)
        # D=torch.where(torch.isinf(D), torch.full_like(D, 0), D)
        # L_sys=torch.eye(D.shape[0])-torch.mm(torch.mm(D,adj),D)
        # values,vector=torch.eig(L_sys,eigenvectors=True)
        # values=values[:,0].view(1,-1)
        print("*****************************")
        adj=g.adj(scipy_fmt='coo')
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj=torch.FloatTensor(adj)
        values,vector=torch.eig(adj,eigenvectors=True)
        values=values[:,0].view(1,-1)
        return values.cuda(),vector.cuda()
    
    def forward(self, x):
        h=x
        vc=self.vector.to(x.device)
        v=self.layers_f(self.values).view(-1,)
        v=self.values.view(-1,)
        v=torch.diag(v)
        L=torch.mm(vc,v)
        L=torch.mm(L,vc.T)
        L=torch.FloatTensor(self.normalize(L.cpu().numpy())).to(x.device)
        for l, layer in enumerate(self.layers):
            h=torch.mm(L,h)
            h = layer(h)
            if l != len(self.layers) - 1:
                if self.activation!=None:
                    h = self.activation(h)
                if self.dropout!=None:
                    h = self.dropout(h)
        return h

class CNN(Net_base):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
    

class CNN_Casual(Net_base):
    def __init__(self):
        super(CNN_Casual, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.mask_matrix = nn.Linear(28, 28, bias=False)
    
    def get_mask_matrix(self):
        #sign不好使
        #return torch.sigmoid(self.state_dict()['mask_matrix.weight'])#.detach().clone())
        return torch.tanh(self.state_dict()['mask_matrix.weight'])#.detach().clone())
        #return torch.relu(self.state_dict()['mask_matrix.weight'])#.detach().clone())

    def forward(self, x):
        for i in range(x.shape[0]):
            for channel in range(x[i].shape[0]):
                x[i][channel]=x[i][channel] * self.get_mask_matrix()
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

if __name__ == "__main__":
    '''
    args=get_args()
    model = CNN()#args.input_size, args.hidden_size, args.output_size, F.relu)
    print(model)
    print(model.state_dict())
    import torchvision
    data=torchvision.datasets.MNIST('./MNIST/',train=True,download=True,
                        transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    x=torch.unsqueeze(data[0][0],0).request_grad(True)
    y=model(x)
    print(x.grad)
    
    '''
    '''
    a=torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
    b=torch.FloatTensor([[1,0,1],[1,0,1],[1,1,1]])
    print(a*b)
    print(a @ b)
    '''
    from dgl.data import CoraGraphDataset as CoraGraphDataset
    args=get_args()
    data_source = CoraGraphDataset()
    model = GCN_Filter(data_source[0].to(args.device),args.input_size, args.hidden_size, args.output_size, F.relu, 0.3)
