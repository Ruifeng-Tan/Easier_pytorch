# For_easier_torch

Until now, all the code are coded by TRF and LGH who are undergraduates of Wuhan University now.

# 1 Construct Graph

## 1.1 sliding window graph

The graph constructing strategy was proposed in TextING, considering the initial code is implemented with Tensorflow, we reimplement it using Pytorch.

[paper_link](https://www.aclweb.org/anthology/2020.acl-main.31.pdf)

```python
def sliding_window(p_l,window_size):
    '''
    construct the graph according to https://www.aclweb.org/anthology/2020.acl-main.31.pdf
    :param p_l: the size of the adjacent matrix
    :param window_size: the sliding window size
    :return: a adjacent matrix that has shape [1,p_l,p_l]
    '''
    adjacent_matrix = p_l.new_zeros(p_l, p_l)
    if p_l > window_size:
        for k in range(-window_size, window_size + 1):
            adjacent_matrix += torch.diag(p_l.new_ones(p_l - abs(k)), k)
    else:
        adjacent_matrix = p_l.new_ones(p_l, p_l)
    adjacent_matrix = adjacent_matrix.unsqueeze(0)
    return adjacent_matrix
```

## 1.2 construct edge index from adjacent matrix batch

You may offer a batch of adjacent matrix, then you get the edge_index you need for the [pytorch_geometric lib](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

```python
def transform_adjacent_matrix(self, edge_index,as_data=True):
    '''
    transform the adjacent matrix into edge_index required by pytorch_geometric
    :param edge_index: [B,N,N]
    :return: [2,edge_num]
    '''
    N = edge_index.size(1)

    graph_index, src, dst = torch.nonzero(edge_index, as_tuple=True)

    edge_index = torch.vstack((src, dst))
    edge_index = edge_index + graph_index * N

    return edge_index
```

# 2 tensor operation

```python
def adjust_tensor_value(adjust_tensor,lengths,filter_value):
    '''
    adjust tenosr value according to the lengths, in last dim, all the value of index bigger than according length will be replaced by filter_value
    :param adjust_tensor: batch tensor of any shape [B,*,H]
    :param lengths: 1-D LongTensor. indicate the adjust begining dim for each tensor in batch tensor a. [B]
    :param filter_value: the replace value
    :return: adjusted tensor
    '''
    packed_a = pack_padded_sequence(adjust_tensor.transpose(1, 2), lengths, batch_first=True, enforce_sorted=False)
    padded_a = pad_packed_sequence(packed_a, batch_first=True,padding_value=filter_value)[0]
    padded_a = padded_a.transpose(1, 2)
    p1d = (0, adjust_tensor.size(-1) - torch.max(lengths))
    padded_a = pad(padded_a, p1d, 'constant', filter_value)
    return padded_a
```