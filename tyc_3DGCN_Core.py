
import torch.nn as nn
import torch as th
from tyc_3DGCN_Block import tyc_3DGCN_Block
import torch.nn.functional as F
from dgl.heterograph import DGLGraph
import dgl
import tyc_batch_norm


class tyc_3DGCN_Core(nn.Module):
    def __init__(self,
               n_tasks: int,
               predictor_hidden_feats: int = 16,
               predictor_dropout: float = 0.,
               bias=True,
               mode: str = 'regression',
               number_atom_features: int = 30,
               n_classes: int = 2,
               
               nfeat_name_list=["x","v","pos"] ):
        super(tyc_3DGCN_Core,self).__init__()


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
        #self.batch_norm=tyc_batch_norm.tyc_batch_norm(number_atom_features=number_atom_features,affine=False)
        self.tyc_3dgcn_module_1=tyc_3DGCN_Block(bias=bias,number_atom_features=number_atom_features)
        #self.tyc_3dgcn_module_2=tyc_3DGCN_Block(bias=bias,number_atom_features=number_atom_features)
        #self.tyc_3dgcn_module_3=tyc_3DGCN_Block(bias=bias,number_atom_features=number_atom_features)
        #self.tyc_3dgcn_module_4=tyc_3DGCN_Block(bias=bias,number_atom_features=number_atom_features)
        

        self.edge_weighting_s=nn.Sequential(nn.Linear(number_atom_features, 1), nn.Sigmoid())
        self.edge_weighting_v=nn.Sequential(nn.Linear(number_atom_features, 1), nn.Sigmoid())


        self.twolayer_linear_s= nn.Sequential(
            nn.Dropout(predictor_dropout),
            nn.Linear(number_atom_features, predictor_hidden_feats),
            
            #nn.Dropout(predictor_dropout),
            #nn.Linear(number_atom_features, predictor_hidden_feats),
            nn.ReLU(),
            ##nn.BatchNorm1d(16),
            #nn.Linear(predictor_hidden_feats, 3*predictor_hidden_feats)
        )
        self.twolayer_linear_v= nn.Sequential(
            nn.Dropout(predictor_dropout),
            nn.Linear(3*number_atom_features, predictor_hidden_feats),
            #nn.Dropout(predictor_dropout),
            #nn.Linear(3*number_atom_features, predictor_hidden_feats),
            nn.ReLU(),
            ##nn.BatchNorm1d(16),
            #nn.Linear(predictor_hidden_feats, 3*predictor_hidden_feats)
        )
        self.onelayer_linear=nn.Sequential(
            nn.Dropout(predictor_dropout),
            nn.Linear(2*predictor_hidden_feats, out_size) ##this line is for 3d included
            #nn.Dropout(predictor_dropout),
            #nn.Linear(4*predictor_hidden_feats, out_size)
        )

        

    def forward(self,g:DGLGraph):
        feats=g.ndata[self.nfeat_name_list[0]]#s
        feats_3d=g.ndata[self.nfeat_name_list[1]]#vector feature
        pos=g.ndata[self.nfeat_name_list[2]]#pos information of molecules
        src,dst=g.edges()

        feats,feats_3d=self.tyc_3dgcn_module_1(feats,feats_3d,src,dst,pos)
        #feats,feats_3d=self.batch_norm(feats,feats_3d)

        #feats,feats_3d=self.tyc_3dgcn_module_2(feats,feats_3d,src,dst,pos)
        #feats,feats_3d=self.tyc_3dgcn_module_3(feats,feats_3d,src,dst,pos)
        #feats,feats_3d=self.tyc_3dgcn_module_4(feats,feats_3d,src,dst,pos)

        #scalar_readout=self.readout(g,feats)# read out from updated scalar features
        #vec_readout=self.readout(g,feats_3d)# read out from updated vector features
        

        with g.local_scope():
            g.edata["h"] = feats
            g.edata["w"] = self.edge_weighting_s(g.edata["h"])
            scalar_readout = dgl.sum_edges(g, feat="h", weight="w")
            g.edata["h"] = feats_3d
            g.edata["w"] = self.edge_weighting_v(g.edata["h"])
            vec_readout = dgl.sum_edges(g, feat="h", weight="w")
            
        
        after_2_layer_s=self.twolayer_linear_s(scalar_readout)
        batch_size=vec_readout.shape[0]
        vec_readout=vec_readout.view(batch_size,-1)
        after_2_layer_v=self.twolayer_linear_v(vec_readout)
        after_2_layer=th.cat([after_2_layer_s,after_2_layer_v],dim=1)
        out=self.onelayer_linear(after_2_layer)
            
        #pre_linear=th.cat([scalar_readout,vec_readout.view(vec_readout.shape[0],-1)],dim=1)
        #out=self.onelayer_linear(pre_linear)


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
        




        return 





        
