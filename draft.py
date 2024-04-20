import torch
import torch.nn as nn

import dgl
import torch
# Define the edges
src_nodes = torch.tensor([3, 1, 2])  # Source nodes for each edge
dst_nodes = torch.tensor([1, 0, 3])  # Destination nodes for each edge
pos=torch.tensor([[2,5,6],[7,8,9],[4,6,3],[4,1,3]])
r=pos[src_nodes[:],:]-pos[dst_nodes[:],:]
print(r)
linear=nn.Linear(10,6)
# Create the DGLGraph
g = dgl.graph((src_nodes, dst_nodes))
node_feats=torch.tensor([[[2,3,4,3,5],[1,2,3,2,2],[4,3,2,4,0]],
                         [[2,3,4,1,3],[4,2,3,6,2],[4,3,2,9,1]],
                         [[1,3,6,1,5],[1,2,3,2,7],[4,3,2,4,6]],
                         [[5,3,4,1,6],[8,2,3,2,2],[4,3,8,4,1]]]).float()

# Print some information about the graph
print(f"Number of nodes: {g.num_nodes()}")
print(f"Number of edges: {g.num_edges()}")
print(f"Edges: {g.edges()}")
src,dest=g.edges()
pre_linear=torch.cat((node_feats[src[:]],node_feats[dest[:]]),dim=2)
print(node_feats)
print(pre_linear)



"""
# Example sizes for demonstration purposes
model=nn.Linear(3,6)
N = 4
features = torch.tensor([[1,2,4,5,3],[0,7,3,2,2],[9,3,5,3,2],[1,2,4,3,6]]).float()
pos = torch.tensor([[1,2],[0,3],[8,9],[3,7]]).float()

# Perform tensor product without using an explicit for loop
result = torch.bmm(pos.unsqueeze(2),features.unsqueeze(1))

# Check the resulting shape
print(pos.shape)
print(pos)
print(features.shape)
print(features)
print(result.shape)
print(result)

"""


"""
result_viewed=result.view(-1,3)
linear_output_viewed=model(result_viewed)
linear_output_viewed=linear_output_viewed.view(4,-1)
linear_output=torch.cat([model(result[:,0,:]),model(result[:,1,:])],dim=1)
print(linear_output_viewed)
print(linear_output)
"""


