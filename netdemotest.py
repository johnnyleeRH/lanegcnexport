import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime


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
        

    def forward(self, index, A, B):
        return slicetensoradd(index, A, B)
        # return scatteradd(index, A, B)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def slicetensoradd(index, A, B):
    indexid = torch.arange(0, index.shape[0], dtype=torch.int64)
    C = torch.cat((index.unsqueeze(0).t(), indexid.unsqueeze(0).t()), dim=1)
    D = torch.unique(C, dim=0)
    uniqindex,count = torch.unique(index, return_counts=True)
    indexcnt = torch.cat((uniqindex.unsqueeze(0).t(), count.unsqueeze(0).t()), dim=1)
    # E = torch.tensor([],device='cuda:0', dtype=torch.int64)
    E = torch.zeros((D[-1][0] + 1, 2), dtype=torch.int64)
    # E = gpu(torch.zeros((D.shape[0], 2), dtype=torch.int64))
    # print(E)
    E[indexcnt[:, 0]] = E[indexcnt[:, 0]].add(indexcnt)
    indexcnt1 = D[E[D[:, 0]][:,1]==1]
    # indexcnt1 = torch.masked_select(D, E[D[:, 0]][:,1]==1)
    # indexcnt1 = torch.index_select(D, dim=0, index=(E[D[:, 0]][:,1]==1))

    # A[indexcnt1[:,0]] = A[indexcnt1[:,0]].add(B[indexcnt1[:,1]])# Warning: ONNX Preprocess - Removing mutation on block inputs. This changes graph semantics.
    newindex = indexcnt1[:,0]
    newB = B[indexcnt1[:,1]]
    A = A.index_put((newindex,), newB, accumulate=True)
    # return A
    # A[indexcnt1[:,0]]
    # A = A.index_put((index,), B, accumulate=True)
    indexcnt2 = D[E[D[:, 0]][:,1]>1]
    for i in indexcnt2: #RuntimeWarning: Iterating over a tensor might cause the trace to be incorrect
        # A[i[0]] = A[i[0]].add(B[i[1]])
        A = A.index_put((i[0],), B[i[1]], accumulate=True)
    return A

def scatteradd(index, A, B):
    newindex = index.unsqueeze(0).t()
    newindex = newindex.repeat(1, A.shape[1])
    return A.scatter_add(0, newindex, B)


def main():
    '''
    A = torch.LongTensor([1,2,1,2,3,4,0])
    B = torch.LongTensor([0,1,2,3,4,5,6])
    C = torch.cat((A.unsqueeze(0).t(), B.unsqueeze(0).t()), dim=1)
    D = torch.unique(C, dim=0)
    index,count = torch.unique(A, return_counts=True)
    indexcnt = torch.cat((index.unsqueeze(0).t(), count.unsqueeze(0).t()), dim=1)
    E = D[indexcnt[D[:, 0]][:,1]==1] # 获取cnt为1的index
    F = D[indexcnt[D[:, 0]][:,1]>1]
    '''
    trained_model = Net()
    index = torch.tensor([1, 3, 2, 3, 3, 1, 4])
    B = torch.tensor([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18], [19,20,21]], dtype=torch.int64)
    A = torch.ones(5,3, dtype=torch.int64)
    backA = A.clone()
    backB = B.clone()
    backindex = index.clone()
    '''
    A = scatteradd(index, A, B)
    # backA.index_add_(0, index, backB)
    # backA[index] = backA[index].add(backB)
    backC = backA.clone()
    backD = backB.clone()
    backC.index_put_((index,), backD, accumulate=True)
    if A.equal(backC):
        print("test equal")
    # return
    # res = trained_model(index, A, B)
    # print(res)
    # return
    '''
    input = (index,A,B)
    with torch.no_grad():
        torch.onnx.export(
                        trained_model,
                        input,
                        "nettest.onnx",
                        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                        opset_version=11)
    ort_session = onnxruntime.InferenceSession("nettest.onnx")
    print('--------------------------------------------------------')
    print(len(ort_session.get_inputs()))
    for i in range(0, len(ort_session.get_inputs())):
        print(ort_session.get_inputs()[i])
    A = backA.clone()
    B = backB.clone()
    index = backindex.clone()

    ort_inputs = {ort_session.get_inputs()[0].name:backindex.cpu().detach().numpy(),
                    ort_session.get_inputs()[1].name:backA.cpu().detach().numpy(),
                    ort_session.get_inputs()[2].name:backB.cpu().detach().numpy()}

    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)

    output = A.index_add(0, index, B)

    np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("ok")
if __name__ == '__main__':
    main()
    