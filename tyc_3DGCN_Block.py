import torch.nn as nn
import torch as th


class tyc_3DGCN_Block(nn.Module):
    def __init__(self,
               bias=True,
               number_atom_features=30,              
               **kwargs):
        super(tyc_3DGCN_Block,self).__init__()
        self.number_atom_features=number_atom_features
        
        #self.linear_s=nn.Sequential(nn.Linear(2*number_atom_features,4*number_atom_features,bias=bias), nn.LeakyReLU(),nn.Linear(4*number_atom_features,1*number_atom_features,bias=bias))
        #self.linear_v=nn.Sequential(nn.Linear(2*number_atom_features,4*number_atom_features,bias=bias), nn.LeakyReLU(),nn.Linear(4*number_atom_features,1*number_atom_features,bias=bias))
        #self.linear_s_1=nn.Sequential(nn.Linear(2*number_atom_features,4*number_atom_features,bias=bias), nn.LeakyReLU(),nn.Linear(4*number_atom_features,1*number_atom_features,bias=bias))
        #self.linear_v_1=nn.Sequential(nn.Linear(2*number_atom_features,4*number_atom_features,bias=bias), nn.LeakyReLU(),nn.Linear(4*number_atom_features,1*number_atom_features,bias=bias))
        self.linear_s=nn.Linear(2*number_atom_features,1*number_atom_features,bias=bias)
        self.linear_v=nn.Linear(2*number_atom_features,1*number_atom_features,bias=bias)
        self.linear_s_1=nn.Linear(2*number_atom_features,1*number_atom_features,bias=bias)
        self.linear_v_1=nn.Linear(2*number_atom_features,1*number_atom_features,bias=bias)

        

    def forward(self,feats,feats_3d,src,dst,pos):

        s_to_s=th.cat((feats[src[:]],feats[dst[:]]),dim=1)
        s_to_s=self.linear_s(s_to_s)#E*F

        v_to_v=th.cat((feats_3d[src[:]],feats_3d[dst[:]]),dim=2)#E*3*2F
        v_to_v=v_to_v.view(-1,2*self.number_atom_features)#3E*2F
        v_to_v=self.linear_v(v_to_v)#3E*F
        v_to_v=v_to_v.view(-1,3,self.number_atom_features)#E*3*F

        r=pos[src[:],:]-pos[dst[:],:]

        s_to_v=th.bmm(r.unsqueeze(2),s_to_s.unsqueeze(1)) #E*3*F
        v_to_s=th.bmm(r.unsqueeze(1),v_to_v) #E*1*F
        v_to_s=v_to_s.squeeze(1)#E*F

        red_s=self.linear_s_1(th.cat((s_to_s,v_to_s),dim=1))#E*F

        red_v=th.cat((v_to_v,s_to_v),dim=2) #E*3*2F
        red_v=red_v.view(-1,2*self.number_atom_features)#3E*2F
        red_v=self.linear_v_1(red_v)#3E*F
        red_v=red_v.view(-1,3,self.number_atom_features)#E*3*F

        return red_s,red_v





        
