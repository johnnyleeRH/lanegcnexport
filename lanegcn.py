# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data import ArgoDataset, collate_fn
from utils import gpu, to_long,  Optimizer, StepLR

from layers import Conv1d, Res1d, Linear, LinearRes, Null
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]

import collections
import pickle


### config ###
config = dict()
"""Train"""
config["display_iters"] = 205942
config["val_iters"] = 205942 * 2
config["save_freq"] = 1.0
config["epoch"] = 0
config["horovod"] = True
config["opt"] = "adam"
config["num_epochs"] = 36
config["lr"] = [1e-3, 1e-4]
config["lr_epochs"] = [32]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])


if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "results", model_name
    )

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 32
config["val_batch_size"] = 32
config["workers"] = 0
config["val_workers"] = config["workers"]


"""Dataset"""
# Raw Dataset
config["train_split"] = os.path.join(
    root_path, "dataset/train/data"
)
config["val_split"] = os.path.join(root_path, "dataset/val/data")
config["test_split"] = os.path.join(root_path, "dataset/test_obs/data")

# Preprocessed Dataset
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(
    root_path, "dataset","preprocess", "train_crs_dist6_angle90.p"
)
config["preprocess_val"] = os.path.join(
    root_path,"dataset", "preprocess", "val_crs_dist6_angle90.p"
)
config['preprocess_test'] = os.path.join(root_path, "dataset",'preprocess', 'test_tmp.p')

"""Model"""
config["rot_aug"] = False
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6
config["n_actor"] = 128
config["n_map"] = 128
config["actor2map_dist"] = 7.0
config["map2actor_dist"] = 6.0
config["actor2actor_dist"] = 100.0
config["pred_size"] = 30
config["pred_step"] = 1
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 6
config["cls_coef"] = 1.0
config["reg_coef"] = 1.0
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2
### end of config ###
def slicetensoradd(index, A, B):
    indexid = gpu(torch.arange(0, index.shape[0], dtype=torch.int64))
    C = gpu(torch.cat((index.unsqueeze(0).t(), indexid.unsqueeze(0).t()), dim=1))
    D = gpu(torch.unique(C, dim=0))
    uniqindex,count = torch.unique(index, return_counts=True)
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
    A[newindex]
    A = A.index_put((newindex,), newB, accumulate=True)
    # return A
    # A[indexcnt1[:,0]]
    # A = A.index_put((index,), B, accumulate=True)
    indexcnt2 = D[E[D[:, 0]][:,1]>1]
    for i in indexcnt2: #RuntimeWarning: Iterating over a tensor might cause the trace to be incorrect
        # A[i[0]] = A[i[0]].add(B[i[1]])
        A = A.index_put((i[0],), B[i[1]], accumulate=True)
    return A

def savepickle(file, input):
    f = open(file, 'wb')
    pickle.dump(input, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def my_index_add(index, A, B):
    # A.index_add_(0, index, B)
    # return A
    # print(index.shape)
    # print("haha--------------------------------------------------------")
    # print("save index")
    # savepickle("./index1.p0", index)
    # print("save A")
    # savepickle("./A1.p0", A)
    # print("save B")
    # savepickle("./B1.p0", B)
    return slicetensoradd(index, A, B)
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
    C = C.index_put((index,), B)
    A = A.index_put_((index,), B, accumulate=True)
    out = A.sub(C)
    return out
    '''


# def my_index_add(index, A, B):
#     # savepickle("./index.p0", index)
#     # savepickle("./A.p0", A)
#     # savepickle("./B.p0", B)
#     A.index_put_((index,), B, accumulate=True)
#     return A


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
'''
def my_index_add(index, A, B):
    new_index=[]
    count = 0
    new_B = []
    index2count = {}
    for i in index:
        if i not in new_index:
            #print(B[count])
            new_index+=[i]
            index2count[i.item()] = len(new_B)
            new_B += [B[count]]
            #new_B.append(B[count])
            #print(new_B)
        else:
            new_B[index2count[i.item()]] = torch.add(new_B[index2count[i.item()]], B[count])
            # new_B[-1]=torch.add(B[count-1],B[count])
            #new_B.pop(-2)
            #print(new_B)
        count = count + 1
    new_index = torch.stack(new_index)
    new_B = torch.stack(new_B)
    A[new_index] = A[new_index].add(new_B)
'''

class Net(nn.Module):
    """
    Lane Graph Network contains following components:
        1. ActorNet: a 1D CNN to process the trajectory input
        2. MapNet: LaneGraphCNN to learn structured map representations 
           from vectorized map data
        3. Actor-Map Fusion Cycle: fuse the information between actor nodes 
           and lane nodes:
            a. A2M: introduces real-time traffic information to 
                lane nodes, such as blockage or usage of the lanes
            b. M2M:  updates lane node features by propagating the 
                traffic information over lane graphs
            c. M2A: fuses updated map features with real-time traffic 
                information back to actors
            d. A2A: handles the interaction between actors and produces
                the output actor features
        4. PredNet: prediction header for motion forecasting using 
           feature from A2A
    """
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)

        self.pred_net = PredNet(config)

    # def forward(self, data:Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[Tensor], List[Tensor]]):# -> Dict[str, List[Tensor]]:
    # expand List[List[Tensor]] => List[Tensor], List[Tensor]...
    def forward(self, data:Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], 
                    List[Tensor], List[Tensor], List[Tensor], List[Tensor],
                    List[Tensor], List[Tensor], List[Tensor], List[Tensor],
                    List[Tensor], List[Tensor]]):
        print("in Net forward")
        # construct actor feature
        actors, actor_idcs = actor_gather(data[0])
        actor_ctrs = data[1]
        # print('*' * 80)
        # print(actors)
        actors = self.actor_net(actors)
        # return actors

        # print(actors)
        # construct map features
        # graph:List[List[Tensor]] = data[2]
        # expand graph
        graph_idcs = data[4]
        graph_ctrs = data[5]
        graph_feats = data[6]
        graph_turn = data[7]
        graph_control = data[8]
        graph_intersect = data[9]
        graph_pre = data[10]
        graph_suc = data[11]
        graph_left = data[12]
        graph_right = data[13]

        nodes, node_idcs, node_ctrs = self.map_net(graph_idcs, graph_ctrs, graph_feats, graph_turn,
                        graph_control, graph_intersect, graph_pre, graph_suc, graph_left, graph_right)
        # return nodes
        # return nodes
        # return node_idcs
        # return nodes => graph_ctrs graph_feats
        # nodes, node_idcs, node_ctrs = self.map_net(graph)
        # return nodes
        # actor-map fusion cycle 
        nodes = self.a2m(nodes, graph_turn, graph_control, graph_intersect, graph_idcs, graph_ctrs, actors, actor_idcs, actor_ctrs)
        # return nodes => graph_ctrs graph_feats graph_turn graph_control graph_intersect
        nodes = self.m2m(nodes, graph_pre, graph_suc, graph_left, graph_right)
        # return nodes ==> graph_ctrs graph_feats graph_turn graph_control graph_intersect
        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        # why no nodes related input ??
        # return actors ==> feats
        actors = self.a2a(actors, actor_idcs, actor_ctrs)
        # return actors ==> feats why not ctrs ??
        # return actors

        # prediction
        # return actors
        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        # return out ==> feats ctrs
        
        rot, orig = data[2], data[3]
        # transform prediction to world coordinates
        # for i in range(len(out["reg"])):
        #     out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
        #         1, 1, 1, -1
        #     )
        for i in range(len(out[1])):
            out[1][i] = torch.matmul(out[1][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
        print("out net forward")
        return out

def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    print("in actor_gather")
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    print("after actor_gather")
    return actors, actor_idcs

# @torch.jit.script
def get_graph_index(str_index:str)->int:
    for i, k1 in enumerate(['idcs', 'ctrs', 'feats', 'turn', 'control', 'intersect', 'pre', 'suc', 'left', 'right']):
        if str_index == k1:
            return i

    for j, k2 in enumerate(['u', 'v']):
        if str_index == k2:
            return j

    raise KeyError(f'no such key {str_index}')

# @torch.jit.script
def graph_get(graph:List[List[Tensor]], indx1:str, indx2:int, indx3:str):
    if indx1 not in ['pre', 'suc']:
        raise RuntimeError('only support pre and suc')

    return graph[get_graph_index(indx1)][indx2 * 2 + get_graph_index(indx3)]

def graph_gather(graphs):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = []
    # graph["idcs"] = node_idcs
    graph.append(node_idcs)
    # graph["ctrs"] = [x["ctrs"] for x in graphs]
    graph.append([x["ctrs"] for x in graphs])

    for key in ["feats", "turn", "control", "intersect"]:
        # graph[key] = torch.cat([x[key] for x in graphs], 0)
        graph.append([torch.cat([x[key] for x in graphs], 0)])

    for k1 in ["pre", "suc"]:
        # graph[k1] = []
        graph.append([])
        for i in range(len(graphs[0]["pre"])):
            # graph[k1].append(dict())
            # graph[-1].append([])
            for k2 in ["u", "v"]:
                # graph[k1][i][k2] = torch.cat(
                #     [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                # )
                graph[-1].append(torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                ))

    for k1 in ["left", "right"]:
        # graph[k1] = dict()
        graph.append([])
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                # x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                x if x.dim() > 0 else graph[get_graph_index("pre")][0].new_empty().resize_(0)
                for x in temp
            ]
            # graph[k1][k2] = torch.cat(temp)
            graph[-1].append(torch.cat(temp))
    return graph

class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, config):
        super(ActorNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 3
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

    # @torch.jit.script
    # def get_lat_res(self, index: int, t: Tensor) -> Tensor:
    #     for i, a_lat in enumerate(self.lateral):
    #         if i == index:
    #             return a_lat(t)

    def forward(self, actors: Tensor) -> Tensor:
        print('actornet forward')
        out = actors

        outputs = []
        for grp in self.groups:
            out = grp(out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            # 上下采样 FPN
            out = F.interpolate(out, scale_factor=2.0, mode="linear", align_corners=False)
            # out += self.lateral[i](outputs[i])
            for j, a_lat in enumerate(self.lateral):
                if i == j:
                    out += a_lat(outputs[i])

        out = self.output(out)[:, :, -1]
        print("after actornet forward")
        return out


def str2int(target:str)->int:
    res = 0
    for ch in target:
        res *= 10
        res += ord(ch) - ord('0')
    return res

class MapNet(nn.Module):
    """
    Map Graph feature extractor with LaneGraphCNN
    """
    def __init__(self, config):
        super(MapNet, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        self.input = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.seg = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def get_fuse_res(self, tensor:torch.Tensor, key:str, i:int):
        for k, v in self.fuse.items():
            if k == key:
                for j, a_v in enumerate(v):
                    if i == j:
                        return a_v(tensor)

        raise RuntimeError(f'module not found with key {key}:{i}')

    # def forward(self, graph:List[List[torch.Tensor]]):
    def forward(self, idcs:List[torch.Tensor], ctrs:List[torch.Tensor], feats:List[torch.Tensor], turn:List[torch.Tensor], control:List[torch.Tensor],
                    intersect:List[torch.Tensor], pre:List[torch.Tensor], suc:List[torch.Tensor], left:List[torch.Tensor], right:List[torch.Tensor]):
        print("in mapnet")
        if (
            len(feats) == 0
            # or len(graph_get(graph, 'pre', -1, "u")) == 0
            # or len(graph_get(graph, "suc", -1, "u")) == 0
            or len(pre[-2]) == 0
            or len(suc[-2]) == 0
        ):
            temp = feats
            return (
                temp[0].new_empty(temp[0].shape).resize_(0),
                [temp[0].new_empty(temp[0].shape).long().resize_(0) for x in idcs],
                [temp[0].new_empty(temp[0].shape).resize_(0)],
            )

        feat = self.input(torch.cat(ctrs, 0))
        feat += self.seg(feats[0])
        feat = self.relu(feat)

        """fuse map"""
        res = feat
        # for i in range(len(self.fuse["ctr"])):
        for k, v in self.fuse.items():
            # temp = self.fuse["ctr"][i](feat)
            if k == 'ctr':
                for i, ctr in enumerate(v):
                    temp = ctr(feat)
                    for key in self.fuse:
                        if key.startswith("pre"):
                            k1 = key[:3]
                            k2 = int(key[3:])
                            a_tmp = self.get_fuse_res(feat[pre[k2 * 2 + 1]], key, i)
                            # tempbase = temp
                            '''
                            temp.index_add_(
                                0,
                                # temp,
                                pre[k2 * 2],
                                # graph[get_index(k1)][get_index(k2)][get_index("u")],
                                # ctr(feat[graph[get_index(k1)][get_index(k2)][get_index("v")]]),
                                a_tmp,
                            )
                            '''
                            # temp = temp.index_put((pre[k2 * 2],), a_tmp, True)
                            temp = my_index_add(pre[k2 * 2], temp, a_tmp)
                            # return temp, idcs, ctrs
                            # if not tempbase.equal(temp):
                            #     print("lrh not equal temp vs tempbase found")
                        if key.startswith("suc"):
                            k1 = key[:3]
                            k2 = int(key[3:])
                            a_tmp = self.get_fuse_res(feat[suc[k2 * 2 + 1]], key, i)
                            # lrhtest = temp.clone()
                            # temp.index_add_(0, suc[k2 * 2], a_tmp)
                            temp = my_index_add(suc[k2 * 2], temp, a_tmp)
                            # A.index_put_((index,), B, accumulate=True)
                            # temp = temp.index_put(tuple(suc[k2 * 2].t()), a_tmp, True)
                            # temp = temp.index_put((suc[k2 * 2],), a_tmp, True)
                            # print([item for item, count in collections.Counter(suc[k2 * 2].tolist()).items() if count > 1])

                            # lrhtest.index_add_(0, suc[k2 * 2], a_tmp)
                            # print(temp[152][1], lrhtest[152][1])
                            # np.testing.assert_allclose(to_numpy(temp), to_numpy(lrhtest), rtol=1e-03, atol=1e-05)
                            # temp.index_add_(0, suc[k2 * 2], a_tmp)
                            # return temp, idcs, ctrs
                            # break

                    if len(left[0] > 0):
                        tmp2 = self.get_fuse_res(feat[left[1]], 'left', i)
                        '''
                        temp.index_add_(
                            0,
                            left[0],
                            # graph[get_graph_index("left")][get_graph_index("u")],
                            tmp2,
                        )
                        '''
                        temp = my_index_add(left[0], temp, tmp2)
                        
                    if len(right[0] > 0):
                        tmp3 = self.get_fuse_res(feat[right[1]], 'right', i)
                        '''
                        temp.index_add_(
                            0,
                            right[0],
                            # graph[get_graph_index("right")][get_graph_index("u")],
                            tmp3,
                        )
                        '''
                        temp = my_index_add(right[0], temp, tmp3)
                        
                    # lrhtest = temp.clone()
                    # my_index_add(right[0], temp, tmp3)
                    # lrhtest.equal(temp)
                    feat = self.get_fuse_res(temp, 'norm', i)
                    feat = self.relu(feat)

                    feat = self.get_fuse_res(feat, 'ctr2', i)
                    feat += res
                    feat = self.relu(feat)
                    res = feat
                    # break # lrh test
        print("after mapnet")
        return feat, idcs, ctrs


class A2M(nn.Module):
    """
    Actor to Map Fusion:  fuses real-time traffic information from
    actor nodes to lane nodes
    """
    def __init__(self, config):
        super(A2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        self.th_ = self.config["actor2map_dist"]

        """fuse meta, static, dyn"""
        self.meta = Linear(n_map + 4, n_map, norm=norm, ng=ng)
        att = []
        for i in range(2):
            att.append(Att(n_map, config["n_actor"]))
        self.att = nn.ModuleList(att)

    def forward(self, feat: Tensor, turn: List[Tensor], control: List[Tensor], intersect: List[Tensor], idcs: List[Tensor], ctrs: List[Tensor], actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        """meta, static and dyn fuse using attention"""
        print("a2m forward")
        meta = torch.cat(
            (
                turn[0],
                control[0].unsqueeze(1),
                intersect[0].unsqueeze(1),
            ),
            1,
        )
        feat = self.meta(torch.cat((feat, meta), 1))

        # for i in range(len(self.att)):
        for a_att in self.att:
            feat = a_att(
                feat,
                idcs,
                ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.th_,
            )
        print("after a2m forward")
        return feat


class M2M(nn.Module):
    """
    The lane to lane block: propagates information over lane
            graphs and updates the features of lane nodes
    """
    def __init__(self, config):
        super(M2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)


    def get_fuse_res(self, key:str, index:int, feat:Tensor):
        for k, v in self.fuse.items():
            if k == key:
                for i, a_v in enumerate(v):
                    if i == index:
                        return a_v(feat)

        raise RuntimeError(f'no module found for key {key}:{index}')

    def forward(self, feat: Tensor, pre: List[Tensor], suc: List[Tensor], left: List[Tensor], right: List[Tensor]) -> Tensor:
        """fuse map"""
        print("m2m forward")
        res = feat
        # for i in range(len(self.fuse["ctr"])):
        for k, v in self.fuse.items():
            if k == 'ctr':
                for i, a_v in enumerate(v):
                    temp = self.get_fuse_res("ctr", i, feat)
                    for key in self.fuse:
                        if key.startswith("pre"):
                            k1 = key[:3]
                            k2 = int(key[3:])
                            tmp = self.get_fuse_res(key, i, feat[pre[k2 * 2 + 1]])
                            '''
                            temp.index_add_(
                                0,
                                pre[k2 * 2],
                                # graph_get(graph, k1, k2, "u"),
                                # self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                                tmp,
                            )
                            '''
                            temp = my_index_add(pre[k2 * 2], temp, tmp)
                        if key.startswith("suc"):
                            k1 = key[:3]
                            k2 = int(key[3:])
                            tmp = self.get_fuse_res(key, i, feat[suc[k2 * 2 + 1]])
                            '''
                            temp.index_add_(
                                0,
                                suc[k2 * 2],
                                # graph_get(graph, k1, k2, "u"),
                                # self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                                tmp,
                            )
                            '''
                            temp = my_index_add(suc[k2 * 2], temp, tmp)
                            

                    if len(left[0] > 0):
                        tmp = self.get_fuse_res('left', i, feat[left[1]])
                        '''
                        temp.index_add_(
                            0,
                            left[0],
                            # graph[get_graph_index("left")][get_graph_index("u")],
                            # self.fuse["left"][i](feat[graph["left"]["v"]]),
                            tmp,
                        )
                        '''
                        temp = my_index_add(left[0], temp, tmp)
                        
                    if len(right[0] > 0):
                        tmp = self.get_fuse_res('right', i, feat[right[1]])
                        '''
                        temp.index_add_(
                            0,
                            right[0],
                            # graph[get_graph_index("right")][get_graph_index("u")],
                            # self.fuse["right"][i](feat[graph["right"]["v"]]),
                            tmp,
                        )
                        '''
                        temp = my_index_add(right[0], temp, tmp)
                        

                    # feat = self.fuse["norm"][i](temp)
                    feat = self.get_fuse_res('norm', i, temp)
                    feat = self.relu(feat)

                    # feat = self.fuse["ctr2"][i](feat)
                    feat = self.get_fuse_res("ctr2", i, feat)
                    feat += res
                    feat = self.relu(feat)
                    res = feat
        print("after m2m forward")
        return feat


class M2A(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """
    def __init__(self, config):
        super(M2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]
        self.th_ = self.config["map2actor_dist"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_map))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor], nodes: Tensor, node_idcs: List[Tensor], node_ctrs: List[Tensor]) -> Tensor:
        print("m2a forward")
        for a_att in self.att:
            actors = a_att(
                actors,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.th_,
            )
        print("after m2a forward")
        return actors


class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """
    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]
        self.th_ = self.config["actor2actor_dist"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_actor))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        print("a2a forward")
        # a_att = self.att[0]
        for a_att in self.att:
            actors = a_att(
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.th_,
            )
        print("after a2a forward")
        return actors


class EncodeDist(nn.Module):
    def __init__(self, n, linear=True):
        super(EncodeDist, self).__init__()
        norm = "GN"
        ng = 1

        block = [nn.Linear(2, n), nn.ReLU(inplace=True)]

        if linear:
            block.append(nn.Linear(n, n))

        self.block = nn.Sequential(*block)

    def forward(self, dist):
        x, y = dist[:, :1], dist[:, 1:]
        dist = torch.cat(
            (
                torch.sign(x) * torch.log(torch.abs(x) + 1.0),
                torch.sign(y) * torch.log(torch.abs(y) + 1.0),
            ),
            1,
        )

        dist = self.block(dist)
        return dist


class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """
    def __init__(self, config):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        self.num_mods = self.config['num_mods']

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        print("prednet forward")
        preds = []
        for a_pred in self.pred:
            preds.append(a_pred(actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)
        # return reg

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        # return feats ok
        cls = self.cls(feats).view(-1, self.num_mods)

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        # return sort_idcs
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)
        # return reg ok but warning
        # out:Dict[str, List[Tensor]] = {'cls':[], 'reg':[]}
        out:List[List[Tensor]] = [[], []]
        # out["cls"] = []
        # out["reg"] = []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            # ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            # out["cls"].append(cls[idcs])
            # out["reg"].append(reg[idcs])
            out[0].append(cls[idcs])
            out[1].append(reg[idcs])
        print("after prednet forward")
        return out


class Att(nn.Module):
    """
    Attention block to pass context nodes information to target nodes
    This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """
    def __init__(self, n_agt: int, n_ctx: int) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agts: Tensor, agt_idcs: List[Tensor], agt_ctrs: List[Tensor], ctx: Tensor, ctx_idcs: List[Tensor], ctx_ctrs: List[Tensor], dist_th: float) -> Tensor:
        print("att forward")
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0

        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= dist_th

            idcs = torch.nonzero(mask)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])

        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        # agtstest = agts.clone()
        # agts.index_add_(0, hi, ctx)
        agts = my_index_add(hi, agts, ctx)
        # for i in range(agts.shape[0]):
        #     if not agts[i].equal(agtstest[i]):
        #         print("************************ not equal,", i)
        # if not agts.equal(agtstest):
        #     print("--------------------not qual lrh")

        # agts = agtstest
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        print("after att forward")
        return agts


class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts


class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out,data):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print(
            "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
            % (loss, cls, reg, ade1, fde1, ade, fde)
        )
        print()


def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs


def get_model():
    net = Net(config)
    net = net.cuda()

    loss = Loss(config).cuda()
    post_process = PostProcess(config).cuda()

    params = net.parameters()
    opt = Optimizer(params, config)


    return config, ArgoDataset, collate_fn, net, loss, post_process, opt
