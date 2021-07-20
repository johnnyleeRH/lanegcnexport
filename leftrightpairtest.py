import numpy as np
import torch

def preprocess(lane_idcs, ctrs, feats, pre_pairs, suc_pairs, left_pairs):
    cross_dist = 6
    left = dict()

    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = ctrs.unsqueeze(1) - ctrs.unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    # shape 558009=747*747
    # hi [0,...,1,...,2,...,746...]
    # wi [0,1,2,..,746, ..., 0,1,2,...746]
    # row_idcs [0,1,2,..,746]
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)


    pre = pre_pairs.new().float().resize_(num_lanes, num_lanes).zero_()
    pre[pre_pairs[:, 0], pre_pairs[:, 1]] = 1
    suc = suc_pairs.new().float().resize_(num_lanes, num_lanes).zero_()
    suc[suc_pairs[:, 0], suc_pairs[:, 1]] = 1

    pairs = left_pairs
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        # 连通性
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = feats[ui]
        f2 = feats[vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)
    print(left)
    return left

    


if __name__ == '__main__':
    lane_ids = {}
    lane_ids['0'] = np.asarray([[10, 0], [11, 0], [12,0], [13, 0], [14, 0]])
    lane_ids['1'] = np.asarray([[10, 1], [11, 1], [12, 1], [13, 1], [14, 1]])
    lane_ids['2'] = np.asarray([[15, 1], [16, 1], [17, 1], [18, 1], [19, 1]])
    lane_ids['3'] = np.asarray([[5, 0], [6, 0], [7,0], [8, 0], [9, 0]])

    ctrs = []
    feats = []
    for lane_id in lane_ids:
        print(lane_id)
        lane = lane_ids[lane_id]
        ctrs.append(np.asarray((lane[:-1] + lane[1:]) / 2.0, np.float32))
        feats.append(np.asarray(lane[1:] - lane[:-1], np.float32))
    graph = {}
    graph['lane_idcs'] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    pre_pairs = []
    pre_pairs.append([0, 3])
    pre_pairs.append([2, 1])
    suc_pairs = []
    suc_pairs.append([1, 2])
    suc_pairs.append([3, 0])
    left_pairs = []
    left_pairs.append([0, 1])

    pre_pairs = torch.from_numpy(np.asarray(pre_pairs, np.int64))
    suc_pairs = torch.from_numpy(np.asarray(suc_pairs, np.int64))
    left_pairs = torch.from_numpy(np.asarray(left_pairs, np.int64))
    # ctrs= torch.tensor([[10, 0], [11, 0], [12,0], [13, 0], [14, 0], [10, 1], [11, 1], [12, 1], [13, 1], [14, 1], [15, 1], [16, 1], [17, 1], [18, 1], [19, 1], [5, 0], [6, 0], [7,0], [8, 0], [9, 0]])

    node_idcs = []
    count = 0
    for i, ctr in enumerate(ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        count += len(ctr)
    lane_idcs = []
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int64))
    lane_idcs = np.concatenate(lane_idcs, 0)
    ctrs = np.concatenate(ctrs, 0)
    feats = np.concatenate(feats, 0)
    ctrs = torch.from_numpy(ctrs)
    feats = torch.from_numpy(feats)
    preprocess(lane_idcs, ctrs, feats, pre_pairs, suc_pairs, left_pairs)
