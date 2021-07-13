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
sys.path.append('/home/johnny/github/lanegcnexport/')
from importlib import import_module

import torch, onnx, collections
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
import onnxruntime


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="test", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="/home/johnny/github/lanegcnexport/actor.36.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)

def main():
    args = parser.parse_args()
    model = import_module(args.model)
    config, _, collate_fn, net, loss, post_process, opt = model.get_model()
    ckpt_path = args.weight
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()
    
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
        # dict_keys(['orig', 'theta', 'rot', 'feats', 'ctrs', 'graph', 'argo_id', 'city'])
        data = dict(data)
        results = []
        with torch.no_grad():
            data['feats'] = gpu(data['feats'])
            data['ctrs'] = gpu(data['ctrs'])
            data['graph'] = graph_gather(to_long(gpu(data['graph'])))
            data['rot'] = gpu(data['rot'])
            data['orig'] = gpu(data['orig'])
            new_data = (data['feats'], data['ctrs'], data['rot'], data['orig'],
                        data['graph'][0], data['graph'][1], data['graph'][2], data['graph'][3],
                        data['graph'][4], data['graph'][5], data['graph'][6], data['graph'][7],
                        data['graph'][8], data['graph'][9])
            test_data = {'feats':data['feats'], 'ctrs':data['ctrs'], 'rot':data['rot'], 'orig':data['orig'],
                        'graph_idcs':data['graph'][0], 'graph_ctrs':data['graph'][1], 'graph_feats':data['graph'][2],
                        'graph_turn':data['graph'][3], 'graph_control':data['graph'][4], 'graph_intersect':data['graph'][5],
                        'graph_pre':data['graph'][6], 'graph_suc':data['graph'][7], 'graph_left':data['graph'][8],
                        'graph_right':data['graph'][9]}
            #torch.onnx.export(net, (new_data,), '36.onnx', opset_version=11, verbose=True)
            # output = {}
            # for i in range(1):
            #     ts = time.time()
            #     output = net(new_data)
            #     ts1 = time.time()
            #     print(f'\n{i + 1} run take {ts1 - ts}')
            # output = net(new_data)
            # torch.onnx.export(scp_mod, new_data, 'test.onnx', example_outputs=output)
            print("beg export")
            # time.sleep(30)
            # return
            # torch.onnx.export(net, (new_data, {}), 'test.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX, opset_version=11,
            #     input_names=['feats', 'ctrs', 'rot', 'orig',
            #     'graph_idcs', 'graph_ctrs', 'graph_feats', 'graph_turn',
            #     'graph_control', 'graph_intersect', 'graph_pre', 'graph_suc', 'graph_left', 'graph_right'])
            # print("after export")
            # print(f'second run take {time.time() - ts1}')
            # scp_mod.save('37.pt')
            # ort_session = onnxruntime.InferenceSession("test.onnx")

            # compute ONNX Runtime output prediction
            # print('--------------------------------------------------------')
            # print(len(ort_session.get_inputs()))
            # for i in range(0, len(ort_session.get_inputs())):
            #     print(ort_session.get_inputs()[i])

            # return
            output = net(new_data)
            # output = script_mod(new_data)
            # print(output)
            results = [x[0:1].detach().cpu().numpy() for x in output[1]]
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
    return

if __name__ == "__main__":
    main()