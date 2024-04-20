from dgllife.model.gnn import GCN
import torch.nn as nn
import torch as th

class tyc_easy_3DGCN_Block(nn.Module):
    def __init__(self,
               graph_conv_layers: list = None,
               activation=None,
               residual: bool = True,
               batchnorm: bool = False,
               dropout: float = 0.,
               number_atom_features=30,
               **kwargs):
        super(tyc_easy_3DGCN_Block,self).__init__()
        if graph_conv_layers==None:
            self.graph_conv_layers=[number_atom_features]
        else:
            self.graph_conv_layers=graph_conv_layers
           
            
        
        self.model_s=GCN(number_atom_features, self.graph_conv_layers,batchnorm=[batchnorm]*len(self.graph_conv_layers),
                         dropout=[dropout]*len(self.graph_conv_layers),activation=[activation]*len(self.graph_conv_layers),residual=[residual]*len(self.graph_conv_layers))
        self.model_v=GCN(number_atom_features, self.graph_conv_layers,batchnorm=[batchnorm]*len(self.graph_conv_layers),
                         dropout=[dropout]*len(self.graph_conv_layers),activation=[activation]*len(self.graph_conv_layers),residual=[residual]*len(self.graph_conv_layers))
        #self.model  =GCN(self.graph_conv_layers[-1],self.graph_conv_layers+[number_atom_features],batchnorm=[batchnorm]*(len(self.graph_conv_layers)+1),
                         #dropout=[dropout]*(len(self.graph_conv_layers)+1),activation=[activation]*(len(self.graph_conv_layers)+1),residual=[residual]*(len(self.graph_conv_layers)+1))
        self.model_linear_s=nn.Linear(in_features=2*number_atom_features,out_features=number_atom_features)
        self.model_linear_v=nn.Linear(in_features=2*number_atom_features,out_features=number_atom_features)
        self.number_atom_features=number_atom_features
    
    def forward(self,g,feats,feats_3d,pos:th.Tensor):#--------->for test, not involved feats_3d and r,just test plain 2D GCN, originally (self,g,feats,feats_3d,pos:th.Tensor)
        s_to_s=self.model_s(g,feats)
        v_to_v=self.model_v(g,feats_3d)

        
        s_to_v=th.bmm(pos.unsqueeze(2),s_to_s.unsqueeze(1))
        v_to_s=th.bmm(pos.unsqueeze(1),v_to_v)
        v_to_s=v_to_s.squeeze(1)


        out_scalar=self.model_linear_s(th.cat([s_to_s,v_to_s],dim=1))
        out_vec=self.model_linear_v(th.cat([v_to_v.view(-1,self.number_atom_features),s_to_v.view(-1,self.number_atom_features)],dim=1))
        out_vec=out_vec.view(-1,3,self.number_atom_features)


        """
        s_to_v=[]
        for i in range(s_to_s.shape[0]):
            if i==0:
                s_to_v=th.matmul(th.transpose(r[i,:].unsqueeze(0),0,1),s_to_s[i,:].unsqueeze(0)).unsqueeze(0)
            else:
                s_to_v=th.cat([s_to_v,th.matmul(th.transpose(r[i,:].unsqueeze(0),0,1),s_to_s[i,:].unsqueeze(0)).unsqueeze(0)])
        v_to_s=[]
        for i in range(v_to_v.shape[0]):
            if i==0:
                v_to_s=th.matmul(r[i,:].unsqueeze(0),v_to_v[i,:,:])
            else:
                v_to_s=th.cat([v_to_s,th.matmul(r[i,:].unsqueeze(0),v_to_v[i,:,:])])
        red_vec=[]
        for j in range(s_to_v.shape[0]):
            input_to_linear=th.cat([s_to_v[j,:,:],v_to_v[j,:,:]])#6*11 
            input_to_linear=th.transpose(input_to_linear,0,1)#11*6
            output_from_linear=self.model_linear_1(input_to_linear)#11*3
            output_from_linear=th.transpose(output_from_linear,0,1)#3*11
            if j==0:
                red_vec=output_from_linear.unsqueeze(0)
            else:
                red_vec=th.cat([red_vec,output_from_linear.unsqueeze(0)])#cat them to N*3*11
        red_scalar=[]
        for m in range(v_to_s.shape[0]):
            input_to_linear=th.stack([v_to_s[m,:],s_to_s[m,:]])#2*11
            input_to_linear=th.transpose(input_to_linear,0,1)#11*2
            output_from_linear=self.model_linear_2(input_to_linear)#11*1
            output_from_linear=th.transpose(output_from_linear,0,1)#1*11
            if m==0:
                red_scalar=output_from_linear
            else:
                red_scalar=th.cat([red_scalar,output_from_linear])#cat them to N*11,no need to squeeze



        out_scalar=self.model(g,red_scalar)
        out_vec=self.model(g,red_vec)
"""

        return out_scalar,out_vec



        


        print("calling foward in tyc_3DGCN")

"""
    def _prepare_batch(self, batch):
    
        try:
            import dgl
        except:
            raise ImportError('This class requires dgl.')

        inputs, labels, weights = batch
        dgl_graphs = [
            graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
        ]
        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(tyc_3DGCNModule, self)._prepare_batch(([], labels,
                                                               weights))
        return inputs, labels, weights
        """
        #out_scalar=self.model(g,s_to_s)
        