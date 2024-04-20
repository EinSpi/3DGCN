from dgllife.model.gnn import GCN
import torch.nn as nn
import torch.nn.functional as F

class tyc_2DGCN_Block(nn.Module):
    def __init__(self,
               graph_conv_layers: list = None,
               activation=F.relu,
               residual: bool = True,
               batchnorm: bool = False,
               dropout: float = 0.,
               number_atom_features=30,
               **kwargs):
        super(tyc_2DGCN_Block,self).__init__()
        if graph_conv_layers==None:
            self.graph_conv_layers=[number_atom_features]
        else:
            self.graph_conv_layers=graph_conv_layers
           
            
        
        self.model_s=GCN(in_feats=number_atom_features,hidden_feats=self.graph_conv_layers,batchnorm=[batchnorm]*len(self.graph_conv_layers),dropout=[dropout]*len(self.graph_conv_layers),activation=[activation]*len(self.graph_conv_layers),residual=[residual]*len(self.graph_conv_layers))
        self.model_linear_s=nn.Linear(in_features=number_atom_features,out_features=number_atom_features)
        
    
    def forward(self,g,feats):
        s_to_s=self.model_s(g,feats)
        #out_scalar=self.model_linear_s(s_to_s)

        return s_to_s

        