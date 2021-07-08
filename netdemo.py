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

def CustomIndexAdd(A, index, B):
    beg = 0
    end = 0
    indexlist = index.tolist()
    indexcontainer = []
    for i in indexlist:
        end = end + 1
        if i in indexcontainer:
            indexcontainer.clear()
            indexcontainer.append(i)
            A[index[beg:end-1]] = A[index[beg:end-1]].add(B[beg:end-1])
            beg = end - 1                                           
        else:
            indexcontainer.append(i)
    if beg < end:
        A[index[beg:end]] = A[index[beg:end]].add(B[beg:end])

def CustomIndexAddTensor(A, index, B):
    beg = 0
    end = 0
    indexlist = index.tolist()
    indexcontainer = []
    for i in indexlist:
        end = end + 1
        if i in indexcontainer:
            indexcontainer.clear()
            indexcontainer.append(i)
            A[index[beg:end-1]] = A[index[beg:end-1]].add(B[beg:end-1])
            beg = end - 1                                           
        else:
            indexcontainer.append(i)
    if beg < end:
        A[index[beg:end]] = A[index[beg:end]].add(B[beg:end])

def slicetensoradd(index, A, B):
    indexid = gpu(torch.arange(0, index.shape[0], dtype=torch.int64))
    C = gpu(torch.cat((index.unsqueeze(0).t(), indexid.unsqueeze(0).t()), dim=1))
    D = gpu(torch.unique(C, dim=0))
    uniqindex,count = torch.unique(index, return_counts=True, sorted=True)
    indexcnt = torch.cat((uniqindex.unsqueeze(0).t(), count.unsqueeze(0).t()), dim=1)
    # E = torch.tensor([],device='cuda:0', dtype=torch.int64)
    E = gpu(torch.zeros((D[-1][0] + 1, 2), dtype=torch.int64))
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
    print(indexcnt2)
    # test = torch.sum(indexcnt2, dim=0)
    # testuniq = torch.unique(indexcnt2[:, 0], dim=0)
    # testuniq = indexcnt2[:, 0]
    # indextemp = gpu(torch.tensor([0]))
    # testuniq = torch.index_select(indexcnt2, 1, indextemp)
    testtemp = gpu(torch.zeros((indexcnt2[indexcnt2.shape[0] - 1][0] + 1, 2), dtype=torch.int64)) # works good
    testuniq = torch.unique(indexcnt2[:, 0])
    # testuniq = D[E[D[:, 0]][:,1]>1]
    print(testuniq)
    temp = testtemp[testuniq]
    print(temp)
    # torch.sum(B[], dim=0)
    # return A
    for i in indexcnt2: #RuntimeWarning: Iterating over a tensor might cause the trace to be incorrect
        A[i[0]] = A[i[0]].add(B[i[1]])
        # A = A.index_put((i[0],), B[i[1]], accumulate=True)
    return A


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self,  index,  A, B):
        return slicetensoradd(index, A, B)
        indexid = gpu(torch.arange(0, index.shape[0], dtype=torch.int64))
        C = gpu(torch.cat((index.unsqueeze(0).t(), indexid.unsqueeze(0).t()), dim=1))
        D = torch.unique(C, dim=0)
        uniqindex,count = torch.unique(index, return_counts=True)
        indexcnt = torch.cat((uniqindex.unsqueeze(0).t(), count.unsqueeze(0).t()), dim=1)
        E = gpu(torch.zeros(D.shape[0], 2, dtype=torch.int64))
        E[indexcnt[:, 0]] = E[indexcnt[:, 0]].add(indexcnt)
        indexcnt1 = D[E[D[:, 0]][:,1]==1]
        A[indexcnt1[:,0]] = A[indexcnt1[:,0]].add(B[indexcnt1[:,1]])
        # indexcnt2 = D[E[D[:, 0]][:,1]>1]
        # for i in indexcnt2:
        #     A[i[0]] = A[i[0]].add(B[i[1]])
        return
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
        C = C.index_put((index,), B)
        A = A.index_put_((index,), B, accumulate=True)
        out = A.sub(C)
        return out
        # slicetensoradd(index, A, B)
        # CustomIndexAdd(A, index, B)
        #A = A.clone()
        
        # C = D.clone()
        # A.index_put_((index,), B, accumulate=True)
        # indexD = range(A.shape[0])
        # uniq = torch.unique(index, sorted=True).tolist()
        # if len(uniq) < A.shape[0]:
        #     paddingindex = gpu(torch.LongTensor(list(set(indexD)-set(uniq))))
        #     index = torch.cat((index, paddingindex))
        #     paddingval = gpu(torch.zeros(1, B.shape[1]))
        #     paddingval = paddingval.expand(paddingindex.shape[0], B.shape[1])
        #     B = torch.cat((B, paddingval))
            

        # C = A.clone()
        # C = C.index_put((index,), B)
        # A.index_put_((index,), B, accumulate=True)
        # out = A.sub(C)
        # return out
        # A = A.index_put((index,), B, True)
        # A = A.index_add((index,), B, True)
        # A[index] = A[index].add(B)
        # return A

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    # a,b,c = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=False, return_inverse=True, return_counts=True)
    trained_model = Net()
    trained_model.eval()

    f = open("index.p0", 'rb')
    index = gpu(pickle.load(f))
    f.close()
    f = open("A.p0", 'rb')
    A = gpu(pickle.load(f))
    f.close()
    f = open("B.p0", 'rb')
    B = gpu(pickle.load(f))
    f.close()
    # DD = A.clone()
    # print(DD[27][0])
    # for i,j in zip(index.tolist(),range(index.shape[0])):
    #     if i == 27:
    #         # print(B[j][0])
    #         print("------",j)
    # return 

    newA = A.clone()
    newB = B.clone()
    newindex = index.clone()

    # A = A.index_add(0, index, B)
    # for i in range(A.shape[0]):
    #     if not A[i].equal(newA[i]):
    #         print("----not equal ", i)
    # np.testing.assert_allclose(to_numpy(A), to_numpy(newA), rtol=1e-03, atol=1e-05)
    # return
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
    # '''
    D = A.clone()
    D.index_put_((index,), B, accumulate=True)
    print("beg export")
    with torch.no_grad():
        torch.onnx.export(
                        trained_model,
                        input,
                        "net_demo_custom.onnx",
                        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                        opset_version=11)
    print("end export")
    # if D.equal(A):
    #     print("----------------- equal")
    # return
    print('finish convert onnx')
    

    ort_session = onnxruntime.InferenceSession("net_demo_custom.onnx")
    print('--------------------------------------------------------')
    print(len(ort_session.get_inputs()))
    for i in range(0, len(ort_session.get_inputs())):
        print(ort_session.get_inputs()[i])

    f = open("A.p0", 'rb')
    A = pickle.load(f)
    f.close()
    f = open("index.p0", 'rb')
    index = pickle.load(f)
    f.close()
    f = open("B.p0", 'rb')
    B = pickle.load(f)
    f.close()
    # A = D.clone()
    # with torch.no_grad():
    #     A.index_put_((index,), B, accumulate=True)
    ort_inputs = {ort_session.get_inputs()[0].name:index.cpu().detach().numpy(),
            ort_session.get_inputs()[1].name:A.cpu().detach().numpy(),
            ort_session.get_inputs()[2].name: B.cpu().detach().numpy()}

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

    output = newA.index_put((newindex,), newB, accumulate=True)

    np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("ok")
    # print(ort_outs[0][0])
    
if __name__ == '__main__':
    main()
    