import torch.nn as nn
from tyc_2DGCN_Block import tyc_2DGCN_Block
from dgl.nn.pytorch import WeightAndSum
import torch.nn.functional as F




class tyc_2DGCN_Core(nn.Module):
    def __init__(self,
               n_tasks: int,
               graph_conv_layers: list = None,
               activation=F.relu,
               residual: bool = True,
               batchnorm: bool = False,
               dropout: float = 0.,
               predictor_hidden_feats: int = 16,
               predictor_dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features: int = 30,
               n_classes: int = 2,
               nfeat_name_list=["x","v","pos"] ):
                #use nfeat_name to access feat from g, study the dgl rule, find proper value for here to access feats, 3dfeats and r from a g.
        super(tyc_2DGCN_Core,self).__init__()
        if mode not in ['classification', 'regression']:
            raise ValueError("mode must be either 'classification' or 'regression'")
        self.mode = mode
        self.n_tasks=n_tasks
        self.n_classes=n_classes
        self.nfeat_name_list= nfeat_name_list
        self.number_atom_features=number_atom_features
        

        if mode == 'classification':
            out_size = n_tasks * n_classes
        else:
            out_size = n_tasks

            
        self.tyc_gcnmodel_1=tyc_2DGCN_Block(number_atom_features=self.number_atom_features,graph_conv_layers=graph_conv_layers,batchnorm=batchnorm,activation=activation,residual=residual,dropout=dropout)
        self.tyc_gcnmodel_2=tyc_2DGCN_Block(number_atom_features=self.number_atom_features,graph_conv_layers=graph_conv_layers,batchnorm=batchnorm,activation=activation,residual=residual,dropout=dropout)
        #self.tyc_gcnmodel_3=tyc_2DGCN_Block(number_atom_features=self.number_atom_features,graph_conv_layers=graph_conv_layers,batchnorm=batchnorm,activation=activation,residual=residual,dropout=dropout)
        #self.tyc_gcnmodel_4=tyc_2DGCN_Block(number_atom_features=self.number_atom_features,graph_conv_layers=graph_conv_layers,batchnorm=batchnorm,activation=activation,residual=residual,dropout=dropout)


        #maybe here I need 2 readout funcs seperately for vec and scalar features
        self.readout=WeightAndSum(number_atom_features)
        self.twolayer_linear_s= nn.Sequential(
            nn.Dropout(predictor_dropout),
            nn.Linear(number_atom_features, 2*predictor_hidden_feats),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(16),
            nn.Linear(2*predictor_hidden_feats, 6*predictor_hidden_feats)
        )
        
        self.onelayer_linear=nn.Sequential(
            nn.Dropout(predictor_dropout),
            nn.Linear(6*predictor_hidden_feats, out_size) ##this line is for 3d included
            #nn.Linear(3*predictor_hidden_feats, out_size) ##this line is without 3d,just for test
            #maybe some unlinear here
        )

        

    def forward(self,g):
        feats=g.ndata[self.nfeat_name_list[0]]#s

        feats=self.tyc_gcnmodel_1(g=g,feats=feats)
        feats=self.tyc_gcnmodel_2(g=g,feats=feats)
        #feats=self.tyc_gcnmodel_3(g=g,feats=feats)
        #feats=self.tyc_gcnmodel_4(g=g,feats=feats)
        
        scalar_readout=self.readout(g,feats)# read out from updated scalar features
        
        after_2_layer_s=self.twolayer_linear_s(scalar_readout)

        out=self.onelayer_linear(after_2_layer_s)


        #here to adjust task type and tasks.

        if self.mode == 'classification':
            if self.n_tasks == 1:
                logits = out.view(-1, self.n_classes)
                softmax_dim = 1
            else:
                logits = out.view(-1, self.n_tasks, self.n_classes)
                softmax_dim = 2
                proba = F.softmax(logits, dim=softmax_dim)
                return proba, logits
        else:
            return out

        
    
        


