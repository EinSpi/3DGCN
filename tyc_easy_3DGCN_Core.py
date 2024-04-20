import torch.nn as nn
import torch as th
from tyc_easy_3DGCN_Block import tyc_easy_3DGCN_Block
from dgl.nn.pytorch import WeightAndSum
import torch.nn.functional as F




class tyc_easy_3DGCN_Core(nn.Module):
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
        super(tyc_easy_3DGCN_Core,self).__init__()
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

        #self.tyc_3d_gcn_outer_loop=nn.ModuleList()
        #for i in range(tyc_3d_gcn_outer_loop_n):
            #self.tyc_3d_gcn_outer_loop.append(tyc_3DGCNModule(graph_conv_layers=graph_conv_layers,
                                                              #number_atom_features=number_atom_features,
                                                              #batchnorm=batchnorm,
                                                              #dropout=dropout,
                                                              #residual=residual,
                                                              #activation=activation))
        self.tyc_gcnmodel_1=tyc_easy_3DGCN_Block(number_atom_features=self.number_atom_features,graph_conv_layers=graph_conv_layers,batchnorm=batchnorm,activation=activation,residual=residual,dropout=dropout)
        self.tyc_gcnmodel_2=tyc_easy_3DGCN_Block(number_atom_features=self.number_atom_features,graph_conv_layers=graph_conv_layers,batchnorm=batchnorm,activation=activation,residual=residual,dropout=dropout)
        self.tyc_gcnmodel_3=tyc_easy_3DGCN_Block(number_atom_features=self.number_atom_features,graph_conv_layers=graph_conv_layers,batchnorm=batchnorm,activation=activation,residual=residual,dropout=dropout)
        self.tyc_gcnmodel_4=tyc_easy_3DGCN_Block(number_atom_features=self.number_atom_features,graph_conv_layers=graph_conv_layers,batchnorm=batchnorm,activation=activation,residual=residual,dropout=dropout)


        #maybe here I need 2 readout funcs seperately for vec and scalar features
        self.readout_s=WeightAndSum(number_atom_features)
        self.readout_v=WeightAndSum(number_atom_features)
        self.twolayer_linear_s= nn.Sequential(
            nn.Dropout(predictor_dropout),
            nn.Linear(number_atom_features, 2*predictor_hidden_feats),
            nn.ReLU(),
            #nn.BatchNorm1d(16),
            nn.Linear(2*predictor_hidden_feats, 3*predictor_hidden_feats)
        )
        self.twolayer_linear_v= nn.Sequential(
            nn.Dropout(predictor_dropout),
            nn.Linear(number_atom_features, 2*predictor_hidden_feats),
            nn.ReLU(),
            #nn.BatchNorm1d(16),
            nn.Linear(2*predictor_hidden_feats, predictor_hidden_feats)
        )
        self.onelayer_linear=nn.Sequential(
            nn.Dropout(predictor_dropout),
            nn.Linear(6*predictor_hidden_feats, out_size) ##this line is for 3d included
            #nn.Linear(3*predictor_hidden_feats, out_size) ##this line is without 3d,just for test
            #maybe some unlinear here
        )

        

    def forward(self,g):
        feats=g.ndata[self.nfeat_name_list[0]]#s
        feats_3d=g.ndata[self.nfeat_name_list[1]]#vector feature
        pos=g.ndata[self.nfeat_name_list[2]]#pos information of molecules

        feats,feats_3d=self.tyc_gcnmodel_1(g=g,feats=feats,feats_3d=feats_3d,pos=pos)
        feats,feats_3d=self.tyc_gcnmodel_2(g=g,feats=feats,feats_3d=feats_3d,pos=pos)
        feats,feats_3d=self.tyc_gcnmodel_3(g=g,feats=feats,feats_3d=feats_3d,pos=pos)
        feats,feats_3d=self.tyc_gcnmodel_4(g=g,feats=feats,feats_3d=feats_3d,pos=pos)#update both v and s features for some times
        scalar_readout=self.readout_s(g,feats)# read out from updated scalar features
        vec_readout=self.readout_v(g,feats_3d)# read out from updated vector features
        after_2_layer_s=self.twolayer_linear_s(scalar_readout)

        batch_size=vec_readout.shape[0]
        vec_readout=vec_readout.view(-1,self.number_atom_features)
        vec_readout=self.twolayer_linear_v(vec_readout)

        after_2_layer_v=vec_readout.view(batch_size,-1)

        after_2_layer=th.cat([after_2_layer_s,after_2_layer_v],dim=1)
        out=self.onelayer_linear(after_2_layer)


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

        
    
        """
        #perform 3dgcn 2times,may be more?
        feats=g.ndata[self.nfeat_name_list[0]]#s
        feats_3d=g.ndata[self.nfeat_name_list[1]]#v
        r=g.ndata[self.nfeat_name_list[2]]#r

        
        
        feats,feats_3d=self.tyc_gcnmodel_1(g,feats,feats_3d,r)
        feats,feats_3d=self.tyc_gcnmodel_2(g,feats,feats_3d,r)
        #seperately perform graphweise aggregation for vec and scalar features
        #scalar_readout=(batchsize,11)
        #vec_readout=(batchsize,3,11)
        scalar_readout=self.readout(g,feats)
        vec_readout=self.readout(g,feats_3d)
        #feed scalar readout to a 2 layer linear nn, resulting in after_2_layer_s=(batchsize,30)
        after_2_layer_s=self.twolayer_linear_s(scalar_readout)
        #tear the vec readout into 3 slices of size (batchsize,11) and pass them through linear seperately, then cat the 3 results with each shape(batchsize,10)
        #resulting in after_2_layer_v=(batchsize,30)
        after_2_layer_v=th.cat([self.twolayer_linear_v(vec_readout[:,0,:]),self.twolayer_linear_v(vec_readout[:,1,:]),self.twolayer_linear_v(vec_readout[:,2,:])],dim=1)
        #cat the after_2_layer_v and after2 layer_s, resulting in (batchsize,30+30)
        after_2_layer=th.cat([after_2_layer_s,after_2_layer_v],dim=1)
        #pass it through the final linear, make final decision  (batchsize,1)
        out=self.onelayer_linear(after_2_layer)


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

        return out
        """


