
import torch
# import transformers
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import os
from scipy.sparse import identity
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.manifold import TSNE
from torch.optim import SGD,Adam
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
import hiddenlayer as hl
from focal_loss import focal_loss
from functools import partial

def cpu_mid_loss(y_pred,y_true,mid=0,pi=0.1,**kwargs):
    eps = torch.tensor([1e-6], dtype=torch.float).to(device)#np.ones_like(y_true)*1e-6
    #pred:shape:torch.Size([128, 2])
    #true:shape:torch.Size([128])
    y_pred = F.log_softmax(y_pred, dim=1) # softmax
    y_pred = torch.exp(y_pred) 
    
    y_pred = y_pred[:,-1]#gather(1,y_true.view(-1,1))
    y_true=y_true.to(torch.float32)
    pos = y_true * y_pred / torch.max(eps,y_true)
    # The maximum two tensors are compared element by element and each larger element is returned to form a new tensor.
    pos = - torch.log(pos + eps)
    neg = (1-y_true) * y_pred/ torch.max(eps, 1-y_true)
    neg = torch.abs(neg- 1e-2) 
    neg = - torch.log(1 - neg )#+ eps
    return torch.mean(0.5*pos + 0.5*neg)#Returns the mean value of all elements in the input tensor.


mid_loss = partial(cpu_mid_loss,mid = 0,pi=0.1)#(1+FN_RATIO)，Fix the last two parameters
# If the graphics card is available, train with the graphics card
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


import xlrd
print(os.getcwd())
blosum_mat=pd.read_csv("./blosum.csv",header=0,index_col=0)
# pam_mat = pd.read_excel("/home1/saizh/sai/PTM/PAM_250.xlsx",header=0,index_col=0,engine='openpyxl')

def embedding(s):# s is the name of the file, path is the path
    # os.chdir(path)
    
    
    f1 = open(s,'r')
    All = f1.read().splitlines()
    All[0]
    h = len(All)# Number of rows, number of samples
    l = len(All[0])# Number of columns, amino acid sequence length
    B = 'ACDEFGHIKLMNPQRSTVWYX'
    B = list(B)
    All_id=[]

    for i in range(h):# Iterate through each sample
        matrix_code1=[]
        for j in range(l):# Iterate through each sequence
            matrix_code1.append(B.index(All[i][j]))# Find the code of the jth sequence of i samples, and convert the sequence into letters
        All_id.append(matrix_code1)
    one_code = np.array(All_id,dtype=object)
    return one_code

# converted into a two-dimensional form, this form can be used as input to the convolutional neural network
import torch.utils.data as data_utils
xtrain=embedding("./data/Y_train.fasta")# '../input/onetrain'../input/fast-s/fastSulf.txt

ytrain=np.array([1 for i in range(len(xtrain)//2)] + [0 for i in range(len(xtrain)//2)])

xvalid = embedding("./data/Y_test.fasta")
yvalid=np.array([1 for i in range(len(xvalid)//2)] + [0 for i in range(len(xvalid)//2)])


xtrain = xtrain.reshape([-1,1,33])# Convert to cnn input format, (52559, 41*21)
xtrain = xtrain.astype(int)
print(xtrain.shape)
xvalid= xvalid.reshape([-1,1,33])# Convert to cnn input format, (52559, 41*21)
xvalid = xvalid.astype(int)
print(xvalid.shape)

train_dataset = data_utils.TensorDataset(torch.Tensor(xtrain),torch.Tensor(ytrain).long())
test_dataset = data_utils.TensorDataset(torch.Tensor(xvalid),torch.Tensor(yvalid).long())

def load_pretrained_embs(embfile):
        with open(embfile, encoding='utf-8') as f:
                lines = f.readlines()
                items = lines[0].split()

        B = 'ACDEFGHIKLMNPQRSTVWYO'
        B = list(B)
        index = len(B)
        reverse = lambda x: dict(zip(x, range(len(x))))
        id= reverse(B)
        embeddings = np.zeros((len(B), 21))
        
        for line in B:
                embeddings[id[line]]=blosum_mat[line]
                
        return embeddings


word2vec_path = "./blosum.csv"



from torchvision import models
from transformers import AdamW, BertConfig
import torch.nn.functional as F
import math
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

class Attention_1(nn.Module):# Multi-headed Self-Attention Mechanism
    def __init__(self, hidden_size=128*2,num_attention_heads=8):
        super(Attention_1, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        
        self.sigmoid =torch.nn.Sigmoid()
        self.hidden_size=hidden_size

        self.layer_norm = torch.nn.LayerNorm(hidden_size).to(device)

        self.query = nn.Linear(hidden_size, self.hidden_size,bias=False).to(device)
        self.key = nn.Linear(hidden_size, self.hidden_size,bias=False).to(device)
        self.value = nn.Linear(hidden_size, self.hidden_size,bias=False).to(device)
        self.gate = nn.Linear(hidden_size, self.hidden_size).to(device)
        
    def transpose_for_scores(self, x):# Divide the vector into two heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)# [:-1]Left closed and right open not included-1
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

        
    def forward(self, batch_hidden):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # linear
        # key = torch.matmul(batch_hidden, self.weight) # b x len x hidden
        # batch_hidden=self.layer_norm(batch_hidden)
        query =self.query(batch_hidden)
        key = self.key(batch_hidden)
        value =self.key(batch_hidden)
        gate = self.sigmoid(self.gate(batch_hidden))
#         key=batch_hidden
#         query=batch_hidden
#         print(key.shape)
#         print(query.shape)
        # compute attention
        query  = self.transpose_for_scores(query)#batch,num_attention_heads,len,attention_head_size
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        
        outputs = torch.matmul(key,query.transpose(-1, -2)) # b x num_attention_heads*len*len

        attention_scores = outputs  / math.sqrt(self.attention_head_size)#(batch,num_attention_heads,len,len)
        
        attn_scores = F.softmax(attention_scores  , dim=-1)  # 
        

        # For an all-zero vector, -1e32 results in 1/len, -inf is nan, and the extra complement is 0
#         masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # sum weighted sources
        context_layer  = torch.matmul(attn_scores, value)#(batch,num_attention_heads,len,attention_head_size
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()#(batch,n,num_attention_heads,attention_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,1)
        batch_outputs  = context_layer.view(*new_context_layer_shape)#(batch,n,all_head_size)
        # print(gate.shape)#32,33,128
        # print(batch_outputs.shape)#32,33,128,1
        batch_outputs =gate * batch_outputs.squeeze(3)

        batch_outputs = batch_outputs
        batch_outputs =  torch.sum(batch_outputs, dim=1)
        # batch_outputs = batch_outputs[:,0]+batch_outputs[:,-1]

        return batch_outputs, attn_scores

import scipy
sigma = 2.5#2.5
mu = 0.0

max_seq_length=33
P_array = np.zeros(max_seq_length)
ann1_index = 16
for i in range(-16,17,1):
    P_array[ann1_index+i] = scipy.stats.norm(mu,sigma).pdf(i)# Implementing a normally distributed cumulative density function with a window length of 1
P_gauss1_array = P_array
P_gauss1_list= list(P_gauss1_array)


all_P_gauss1_list= torch.tensor(P_gauss1_list, dtype=torch.float).to(device)
all_P_gauss1_list = all_P_gauss1_list.unsqueeze(0)
# all_P_gauss1_list=all_P_gauss1_list.repeat(batch_size,1,1)gelu
print(all_P_gauss1_list.shape)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # The activation function used after the action with guass probability
        self.activation_final = nn.Tanh()#nn.Tanh()
        self.gelu = GELU()

        extword_embed = load_pretrained_embs(word2vec_path)
        
        self.extword_embed = nn.Embedding(21, 21)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))

        self.extword_embed.weight.requires_grad = False

        self.word_embed = nn.Embedding(21, 100)#
        self.word_embed.weight.requires_grad = True
        self.P_gauss1_bs = all_P_gauss1_list

        


        vgg16_bn_=models.vgg16_bn(pretrained=True)# Support our use of convolutional layers in custom architectures for some of the layers 7-13
        resnet18_=models.resnet18(pretrained=True)# Use the residual block part, the depth is not enough to increase the number of feature map, generally take the whole residual block layer3
        
        self.conv1=nn.Sequential(nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1)
                                ,nn.BatchNorm2d(64,1e-5)
                                ,nn.PReLU())
        vgg16_bn_.features[9]=nn.PReLU()
        vgg16_bn_.features[12]=nn.PReLU()
        resnet18_.layer3[0].relu=nn.PReLU()
        resnet18_.layer3[1].relu=nn.PReLU()
        
        self.block2 = vgg16_bn_.features[7:14]# 14*14.2 convolution and 2 BN layers, last max pooling, 64 channels input, 128 channels output
        self.block3 = resnet18_.layer3# 7*7,It is stated in the residual network not to bring bias, see what this resnet18 network structure is
        self.avgpool = resnet18_.avgpool# Convert all feature map dimensions to 1*1
        self.cnn = nn.Sequential(
            self.block2,
            self.block3,
            self.avgpool,
        )
        for param in self.cnn.parameters():
            param.requires_grad = True
        
        self.attention_net=Attention_1()

        self.bilstm1 = torch.nn.LSTM(100, 128, 1, bidirectional=True,batch_first=True)# Input dimension, hidden layer dimension, number of layers
        self.layer_norm = torch.nn.LayerNorm(64*2).to(device)
        # self.layer_norm = torch.nn.LayerNorm(128).to(device)
#         self.fc=nn.Sequential(
#             nn.Dropout(p=0.1),
#             nn.Linear(in_features=256,out_features=10,bias=True)
#         )
        self.fc = nn.Sequential(
            nn.Linear(256*3,256),
            nn.PReLU(),
            nn.Dropout(p=0.5),
     
            nn.Linear(256, 2),
                
        )
    
    
    def forward(self,word_ids):
        # word_ids: sen_num x sent_len
        h_shared = self.word_embed(word_ids.long())  #sen_num x sent_len x21
        x = self.extword_embed(word_ids.long())
        
        x =self.conv1(x)
        x=self.cnn(x)
        # x = self.block3(self.block2(x))
        # x = self.avgpool(x)#64*256*1*1
        
        h_shared = h_shared.squeeze(dim=1)
#         x = x.view(x.shape[0],256,-1)#bacth_size,256,1,1
    
#         x1,_= self.bilstm1(h_shared)
# #  #         (forward_out, backward_out) = torch.chunk(x, 2, dim = 2)
#         x = forward_out + backward_out  #[batch,seq_len, hidden_size]

        #基于gauss的lstm

        output, (final_hidden_state, final_cell_state)  = self.bilstm1(h_shared)

        
        hidden_att ,attn_scores= self.attention_net(output)
        
        
        
        P_gauss1_bs=self.P_gauss1_bs#1,1,33
        P_gauss1_bs=P_gauss1_bs.repeat(x.shape[0],1,1)
        # print(output.shape)#32，33，256
        # print(P_gauss1_bs.shape)#32，1，33
        gauss_entity1=torch.matmul(P_gauss1_bs,output)#32，1，256
        gauss_entity1=gauss_entity1.squeeze(dim=1)
        # batch_outputs =  torch.sum(batch_outputs, dim=1)
        # gauss_entity1 = self.layer_norm(gauss_entity1)
        gauss_entity1 = self.gelu(gauss_entity1)
        
        # hidden_lstm = torch.cat((final_hidden_state[0],final_hidden_state[1]),dim=1)
        # hidden_lstm, attention = self.attention_net(output)
        
        hidden_cnn = x.view(x.shape[0],256)#seq为256
        # hidden=torch.cat((hidden_lstm,hidden_cnn),dim=1)#bs,512

        hidden_lstm = gauss_entity1

        # print(gauss_entity1.shape)#32,256
        
        hidden=torch.cat((hidden_cnn,hidden_lstm,hidden_att),dim=1)
        # hidden =self.gelu(hidden_cnn*hidden_lstm*hidden_att)


        # hidden, attention = self.attention_net(hidden)

        # hidden = hidden.view(hidden.shape[0],256)
        hidden = self.fc(hidden)
        return hidden
# Define the loss function and calculate how much the difference is
# Call the model just defined and transfer the model to the GPU (if available)
model = MyNet().to(device)
print(model)
loss_fn =nn.CrossEntropyLoss()#nn.CrossEntropyLoss()# Cross-entropy loss function focal_loss()#alpha=[0.55,0.45],

# Define optimizers to optimize model parameters during training

optimizer = torch.optim.Adam(model.parameters(),lr=2e-4,weight_decay=1e-4)
#
# CNN:3e-4:,2e-3:,2e-5:
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

if __name__ == '__main__':
#     train_dataset = datasets.MNIST(
#         root='./', train=True, transform=data_tf, download=True)
#     test_dataset = datasets.MNIST(root='./', train=False, transform=data_tf)
    # (Hyper parameters)
    batch_size=32
#     data_tf = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize([0.5], [0.5])])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Define the training function that requires
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

epoch_list=[]
test_loss_list=[]
train_loss_list=[]
train_acc_list=[]
test_acc_list=[]
# writer = SummaryWriter("/kaggle/working")
step=1
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=2e-06)     
   
 #>0.1f means right-aligned, takes up 0 positions, decimal 1 floating point number
# A total of 10 training sessions
epochs = 20
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=2e-06) 


# model.load_state_dict(torch.load("./ST_model/model.pth"))



model.load_state_dict(torch.load("./Y_model/model_Y.pth"),strict=False)
model.eval()
correct = 0
total = 0
i=0
preds_list=[]
decision_score=[]
best_y_true_=[]
best_y_pred_=[]
import torch.nn.functional as F
with torch.no_grad():
    for data in valid_dataloader:
        images, labels = data
        
        if torch.cuda.is_available():
            images,labels= images.cuda(),labels.cuda()
            
        outputs =model(images)
        _, predicted = torch.max(outputs.data, 1)
        outputs.data=F.softmax(outputs.data,dim=1)
        
      
        decision_score.extend(list(outputs.data.cpu().numpy()[:,1]))
        for preds in predicted:
            preds_list.append(preds.cpu().numpy())
            
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
#         print(predicted)
print('Accuracy of the network on the  test images: %f %%' % ( 100 *correct / total))
from sklearn.metrics import * 
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

obj = confusion_matrix(yvalid, preds_list)
tn = obj[0][0]
fp = obj[0][1]
fn = obj[1][0]
tp = obj[1][1]
sp = tn/(tn+fp)
sn = tp/(tp+fn)
mcc= (tp*tn-fp*fn)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
print("sp: ",sp)

# Precision, accuracy, percentage of correct predictions for all sample species

accuracy = accuracy_score(yvalid, preds_list)
print("ACC: ",accuracy)

# # Check accuracy P (accuracy rate), precision (check accuracy rate) = TP/(TP+FP)
precision = precision_score(yvalid, preds_list) # 'micro', 'macro', 'weighted'
print("Precision: ",precision)

# # Check all rate R (recall rate), originally right, the proportion of correct predictions; recall (check all rate) = TP/(TP + FN)
recall = recall_score(yvalid, preds_list) # 'micro', 'macro', 'weighted'
print("SN: ",recall)

# F1-Score
f1 = f1_score(yvalid, preds_list, average='macro')     # 'micro', 'macro', 'weighted'
print("F1 Score: ",f1)

#MCC
mcc= matthews_corrcoef(yvalid, preds_list)
print("mcc",mcc)

from sklearn.metrics import roc_curve, auc
fprs, tprs, thresholds = roc_curve(yvalid, decision_score)
roc_auc = auc(fprs, tprs)
print("auc",roc_auc)

#YDataset

#finetune