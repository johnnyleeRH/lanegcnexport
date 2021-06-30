import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime

import pickle

def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        # for k, v in data.items():
        #     data[k] = gpu(v)
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

    def forward(self,  index,  A, B):
        # D = A.clone()
        A.index_put_((index,), B, accumulate=True)
        return A
        # A = A.index_put_((index,),B,accumulate=True)
        # return A
        # uniq, inverse_indices = torch.unique(index, sorted=True, return_inverse=True)
        # print(uniq)
        # print(inverse_indices)
        # print(index)
        # print(uniq)
        '''
        count = 0
        for i in index:
            A.index_put_((i.unsqueeze(0),), B[count].unsqueeze(0), accumulate=True)
            count = count + 1
        '''

        indexA = range(A.shape[0])
        # uniq = torch.unique(index, sorted=True)
        uniq = torch.unique(index, sorted=True).tolist()
        if len(uniq) < A.shape[0]:
            paddingindex = gpu(torch.LongTensor(list(set(indexA)-set(uniq))))
            index = torch.cat((index, paddingindex))
            paddingval = gpu(torch.zeros(1, B.shape[1]))
            paddingval = paddingval.expand(paddingindex.shape[0], B.shape[1])
            B = torch.cat((B, paddingval))
            
        C = A.clone()
        D = A.clone()
        C = C.index_put((index,), B)
        A.index_put_((index,), B, accumulate=True)
        # out = A.sub(C)
        return A

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    trained_model = Net()

    f = open("index.p0", 'rb')
    index = pickle.load(f)
    f.close()
    f = open("A.p0", 'rb')
    A = pickle.load(f)
    f.close()
    f = open("B.p0", 'rb')
    B = pickle.load(f)
    f.close()
    # A= gpu(torch.ones(5, 3))
    # dim = torch.zeros(1)
    # print(dim)
    # B = gpu(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9],[10, 11, 12], [100, 101, 102]], dtype=torch.float))
    # B = torch.tensor([[1,2,3]], dtype=torch.float)
    # index = torch.tensor([0, 1,2,2]), torch.tensor([0,1,2,2])
    # index = torch.LongTensor([[0,0], [1, 1], [2, 2], [2, 2]])
    # index = gpu(torch.tensor([1, 2, 3, 2, 0]))
    # index = torch.tensor([0,1,2,3,4])
    # index = torch.tensor([0])


    # indexA = range(A.shape[0])
    # uniq = torch.unique(index, sorted=True).tolist()
    # paddingindex = torch.LongTensor(list(set(indexA)-set(uniq)))
    # index = torch.cat((index, paddingindex))
    # paddingval = B[0].clone()
    # paddingval = paddingval.expand(paddingindex.shape[0], B.shape[1])
    # B = torch.cat((B, paddingval))

    input = (index,A,B)
    # res = trained_model(index, A, B)
    # print(res)
    # return
    with torch.no_grad():
        torch.onnx.export(
                        trained_model,
                        input,
                        "net_demo.onnx",
                        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                        opset_version=11)

    print('finish convert onnx')
    ort_session = onnxruntime.InferenceSession("net_demo.onnx")
    print(len(ort_session.get_inputs()))  # 3
    print('onnx input shape:',ort_session.get_inputs()[0].shape) #  [5]
    print('onnx input shape:',ort_session.get_inputs()[1].shape) #  [5,3]
    print('onnx input shape:',ort_session.get_inputs()[2].shape) #  [5,3]

    ort_inputs = {ort_session.get_inputs()[0].name:index.cpu().detach().numpy(),ort_session.get_inputs()[1].name:A.cpu().detach().numpy(),ort_session.get_inputs()[2].name: B.cpu().detach().numpy()}

    ort_outs = ort_session.run(None, ort_inputs)

    f = open("index.p0", 'rb')
    newindex = pickle.load(f)
    f.close()
    f = open("A.p0", 'rb')
    newA = pickle.load(f)
    f.close()
    f = open("B.p0", 'rb')
    newB = pickle.load(f)
    f.close()
    if newindex.equal(index):
        print("index equal")
    if newA.equal(A):
        print("A equal")
    if newB.equal(B):
        print("B equal")

    output = newA.index_put((newindex,), newB, accumulate=True)

    np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)
    # print(ort_outs[0][0])
    
if __name__ == '__main__':
    main()
    