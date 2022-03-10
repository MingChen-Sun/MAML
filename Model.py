import torch
import torch.nn as nn
import torch.nn.functional as F
from Get_Args import get_args

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,activation,dropout=None):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.activation=activation
        self.dropout=dropout
        if len(hidden_size) < 1:
            self.linear = nn.Linear(input_size, output_size)
        else:
            self.layers.append(nn.Linear(input_size, hidden_size[0]))
            for i in range(len(hidden_size)-1):
                self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.layers.append(nn.Linear(hidden_size[-1], output_size))
    
    def get_parameters(self):
        parameters=[]
        for name, param in self.named_parameters():
            parameters.append([name,param.data.clone()])
        return parameters

    def load_parameters(self,fast_parameters):
        for index,param in enumerate(self.parameters()):
            param.data=fast_parameters[index][1]
        
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def get_parameters(self):
        parameters=[]
        for name, param in self.named_parameters():
            parameters.append([name,param.data.clone()])
        return parameters

    def load_parameters(self,fast_parameters):
        for index,param in enumerate(self.parameters()):
            param.data=fast_parameters[index][1]
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
    

class CNN_Casual(nn.Module):
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
        return torch.sigmoid(self.state_dict()['mask_matrix.weight'])#.detach().clone())
        #return torch.tanh(self.state_dict()['mask_matrix.weight'])#.detach().clone())
        #return torch.relu(self.state_dict()['mask_matrix.weight'])#.detach().clone())

    def get_parameters(self):
        parameters=[]
        for name, param in self.named_parameters():
            parameters.append([name,param.data.clone()])
        return parameters

    def load_parameters(self,fast_parameters):
        for index,param in enumerate(self.parameters()):
            param.data=fast_parameters[index][1]
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
    model = Net(args.input_size, args.hidden_size, args.output_size, F.relu)
    print(model)
    print(model.state_dict())
    '''
    a=torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
    b=torch.FloatTensor([[1,0,1],[1,0,1],[1,1,1]])
    print(a*b)
    print(a @ b)
    
