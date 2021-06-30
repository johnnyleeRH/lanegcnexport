# ---------------------------------------------------------------------------
# Learning Lane Graph Representations for Motion Forecasting
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Ming Liang, Yun Chen
# ---------------------------------------------------------------------------

import argparse
import os
import time
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
from importlib import import_module

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lanegcn import graph_gather
from data import ArgoTestDataset
from utils import Logger, load_pretrain, gpu, to_long, half

import vizdata
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import numpy as np

from lanegcn import Net, config

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="angle90", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="val", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)

# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
def main():
    
    data = {}
    f = open("/apollo/inputdatawithpadding/input0.p0", 'rb')
    modeltest = pickle.load(f)
    f.close()
    data['feats'] = []
    for val in modeltest["feats"]:
        data['feats'].append(torch.from_numpy(np.asarray(val, dtype=np.float32)))
    data['ctrs'] = []
    for val in modeltest["ctrs"]:
        data['ctrs'].append(torch.from_numpy(np.asarray(val, dtype=np.float32)))
    data['orig'] = []
    for val in modeltest["origs"]:
        data['orig'].append(torch.from_numpy(np.asarray(val, dtype=np.float32)))
    data['rot'] = []
    for val in modeltest["rots"]:
        data['rot'].append(torch.from_numpy(np.asarray(val, dtype=np.float32)))
    data['graph'] = []
    floattype = ['graph0', 'graph1', 'graph2', 'graph3', 'graph4', 'graph5']
    longtype = ['graph6', 'graph7', 'graph8', 'graph9']
    id = 0
    for key in floattype:
        data['graph'].append([])
        for val in modeltest[key]:
            data['graph'][id].append(torch.from_numpy(np.asarray(val, dtype=np.float32)))
        id = id + 1
    for key in longtype:
        data['graph'].append([])
        for val in modeltest[key]:
            data['graph'][id].append(torch.from_numpy(np.asarray(val, dtype=np.long)))
        id = id + 1

    # script_mod = torch.jit.load('36.pt')
    # scp_mod = script_mod.cuda()
    # scp_mod.eval()
    


    args = parser.parse_args()
    model = import_module(args.model)
    config, _, collate_fn, net, loss, post_process, opt = model.get_model()
    ckpt_path = args.weight
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()

    trace_mod = torch.jit.trace(net, data)
    torch.jit.save(trace_mod, 'lrhtrace.pt')
    return

    # script_mod = torch.jit.script(net)
    # torch.save(script_mod.state_dict(), 'lrh.pt')

    
    # return
    # scp_mod = Net(config)
    # scp_mod = scp_mod.cuda()
    # scp_mod.load_state_dict(torch.load('lrh.pt'))
    # scp_mod.eval()
    '''
    with torch.no_grad():
        data['feats'] = gpu(data['feats'])
        data['ctrs'] = gpu(data['ctrs'])
        data['graph'] = (gpu(data['graph']))
        data['rot'] = gpu(data['rot'])
        data['orig'] = gpu(data['orig'])
        new_data = (data['feats'], data['ctrs'], data['graph'], data['rot'], data['orig'])
        # script_mod = torch.jit.trace(net, new_data)
        # torch.jit.save(script_mod, 'lrh.pt')
        # return
        # for i in range(0, 10):
        ts = time.time()
        # output = scp_mod(new_data)
        output = net(new_data)
        ts1 = time.time()
        #print(len(output['cls']), output['cls'][0].shape)
        # print(f'{i + 1} run take {ts1 - ts}')
    return
    '''

    # Import all settings for experiment.
    # args = parser.parse_args()
    # model = import_module(args.model)
    # config, _, collate_fn, net, loss, post_process, opt = model.get_model()

    # load pretrain model
    # ckpt_path = args.weight
    # if not os.path.isabs(ckpt_path):
    #     ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    # ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    # load_pretrain(net, ckpt["state_dict"])
    # net.eval()
    #script_mod = torch.jit.load('36_151.pt')
    # net.to(torch.device('cpu'))
    #script_mod = torch.jit.script(net)
    #script_mod = script_mod.half()
    #torch.jit.save(script_mod, '36.pt')
    # print('net scripted')
    
    # script_mod = script_mod.cpu()

    # Data loader for evaluation
    dataset = ArgoTestDataset(args.split, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    # begin inference
    preds = {}
    gts = {}
    cities = {}
    for ii, data in tqdm(enumerate(data_loader)):
        data = dict(data)
        results = []
        with torch.no_grad():
            data['feats'] = gpu(data['feats'])
            data['ctrs'] = gpu(data['ctrs'])
            data['graph'] = graph_gather(to_long(gpu(data['graph'])))
            data['rot'] = gpu(data['rot'])
            data['orig'] = gpu(data['orig'])
            new_data = (data['feats'], data['ctrs'], data['graph'], data['rot'], data['orig'])
            # torch.onnx.export(net, (new_data,), '36.onnx', opset_version=11, verbose=True)
            output = {}
            for i in range(1):
                ts = time.time()
                output = net(new_data)
                ts1 = time.time()
                print(f'\n{i + 1} run take {ts1 - ts}')
            # output = scp_mod(new_data)
            # print(f'second run take {time.time() - ts1}')
            # scp_mod.save('37.pt')

            # output = net(new_data)
            # output = script_mod(new_data)
            # print(output)
            results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
        for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], results)):
            preds[argo_idx] = pred_traj.squeeze()
            cities[argo_idx] = data["city"][i]
            gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None

    afl = ArgoverseForecastingLoader("/home/johnny/github/torchexport/LaneGCN-master.bak/dataset/test_obs/tmp/")
    df = afl.get("/home/johnny/github/torchexport/LaneGCN-master.bak/dataset/test_obs/tmp/1.csv").seq_df
    frames = df.groupby("OBJECT_TYPE")
    input_ = np.zeros((20,2), dtype=float)
    gt_ = np.zeros((30,2), dtype=float)
    for group_name, group_data in frames:
        if group_name == 'AGENT':
            input_[:, 0] = group_data['X'][:20]
            input_[:, 1] = group_data['Y'][:20]
            gt_[:, 0] = group_data['X'][20: ]
            gt_[:, 1] = group_data['Y'][20: ]
    #input: (20, 2)
    #preds[1]: (6, 30, 2)
    #gt: none
    #city: str
    vizdata.viz_predictions(input_, preds[1], gt_, cities[1])
    # save for further visualization
    res = dict(
        preds = preds,
        gts = gts,
        cities = cities,
    )
    # torch.save(res,f"{config['save_dir']}/results.pkl")
    
    # evaluate or submit
    if args.split == "val":
        # for val set: compute metric
        from argoverse.evaluation.eval_forecasting import (
            compute_forecasting_metrics,
        )
        # Max #guesses (K): 6
        _ = compute_forecasting_metrics(preds, gts, cities, 6, 30, 2)
        # Max #guesses (K): 1
        _ = compute_forecasting_metrics(preds, gts, cities, 1, 30, 2)
    else:
        # for test set: save as h5 for submission in evaluation server
        from argoverse.evaluation.competition_util import generate_forecasting_h5
        generate_forecasting_h5(preds, f"{config['save_dir']}/submit.h5")  # this might take awhile
    # import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    main()
