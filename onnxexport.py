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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def main():
    '''
    data = {}
    f = open("/apollo/inputdata/16824.p0", 'rb')
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
    data['graph'] = []，转的过程中会有提示
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
    '''


    args = parser.parse_args()
    model = import_module(args.model)
    config, _, collate_fn, net, loss, post_process, opt = model.get_model()
    ckpt_path = args.weight
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()

    '''
    data = {}
    cuda0 = torch.device('cuda:0')
    data['feats'] = torch.zeros((15, 20, 3), device=cuda0)
    data['ctrs'] = torch.zeros((15, 2), device=cuda0)
    data['rot'] = torch.zeros((2, 2), device=cuda0)
    data['orig'] = torch.zeros((2), device=cuda0)
    graph_idcs_list = []
    graph_idcs_list.append(torch.zeros((747), device=cuda0, dtype=torch.long))
    graph_ctrs_list = []
    graph_ctrs_list.append(torch.zeros((747, 2), device=cuda0))
    graph_feats = []
    graph_feats.append(torch.zeros((747, 2), device=cuda0))
    graph_turn_list = []
    graph_turn_list.append(torch.zeros((747, 2), device=cuda0))
    graph_control_list = []
    graph_control_list.append(torch.zeros((747), device=cuda0))
    graph_intersect_list = []
    graph_intersect_list.append(torch.zeros((747), device=cuda0))
    graph_pre_list = []
    graph_suc_list = []
    for i in range(0, 8):
        graph_pre_list.append(torch.zeros((747), device=cuda0, dtype=torch.long))
        graph_suc_list.append(torch.zeros((747), device=cuda0, dtype=torch.long))
    for i in range(0, 2):
        graph_pre_list.append(torch.zeros((726), device=cuda0, dtype=torch.long))
        graph_suc_list.append(torch.zeros((726), device=cuda0, dtype=torch.long))
    for i in range(0, 2):
        graph_pre_list.append(torch.zeros((673), device=cuda0, dtype=torch.long))
        graph_suc_list.append(torch.zeros((673), device=cuda0, dtype=torch.long))
    graph_left_list = []
    graph_right_list = []
    for i in range(0, 2):
        graph_left_list.append(torch.zeros((144), device=cuda0, dtype=torch.long))
        graph_right_list.append(torch.zeros((144), device=cuda0, dtype=torch.long))
    data['graph'] = []
    data['graph'].append(graph_idcs_list)
    data['graph'].append(graph_ctrs_list)
    data['graph'].append(graph_feats)
    data['graph'].append(graph_turn_list)
    data['graph'].append(graph_control_list)
    data['graph'].append(graph_intersect_list)
    data['graph'].append(graph_pre_list)
    data['graph'].append(graph_suc_list)
    data['graph'].append(graph_left_list)
    data['graph'].append(graph_right_list)
    new_data = (data['feats'], data['ctrs'], data['graph'], data['rot'], data['orig'])
    print("beg export")
    torch.onnx.export(net, (new_data,), 'test.onnx', opset_version=11)
    print("after export")
    return
    '''

    # script_mod = torch.jit.script(net)
    # scp_mod = script_mod.cuda()
    # scp_mod.eval()
    # torch.jit.save(script_mod, 'lrh.pt')
    # return

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
            new_data = (data['feats'], data['ctrs'], data['rot'], data['orig'],
                        data['graph'][0], data['graph'][1], data['graph'][2], data['graph'][3],
                        data['graph'][4], data['graph'][5], data['graph'][6], data['graph'][7],
                        data['graph'][8], data['graph'][9])

                #torch.onnx.export(net, (new_data,), '36.onnx', opset_version=11, verbose=True)
            # output = {}
                # for i in range(1):
                #     ts = time.time()
                #     output = net(new_data)
                #     ts1 = time.time()
                #     print(f'\n{i + 1} run take {ts1 - ts}')
            # output = net(new_data) # output tensor [15, 128]
            # return
                # torch.onnx.export(scp_mod, new_data, 'test.onnx', example_outputs=output)
                # print("beg export")
                # time.sleep(30)
                # return
                
            ort_session = onnxruntime.InferenceSession("test.onnx")

            input1 = torch.cat(data['feats'],0)
            input2 = torch.cat(data['ctrs'],0)
            input4 = torch.cat(data['rot'],0)
            input5 = torch.cat(data['orig'],0)
            input3_2 =  torch.cat(data['graph'][1],0).clone()#data['graph'][1][0]
            input3_3 =  torch.cat(data['graph'][2],0).clone()#data['graph'][2][0]
            input3_4 =  data['graph'][3][0]
            input3_5 =  data['graph'][4][0]
            input3_6 =  data['graph'][5][0]
            input3_7 =  data['graph'][6][0].clone()
            input3_8 =  data['graph'][6][1].clone()
            input3_9 =  data['graph'][6][2]
            input3_10 =  data['graph'][6][3]
            input3_11 =  data['graph'][6][4]
            input3_12 =  data['graph'][6][5]
            input3_13 =  data['graph'][6][6]
            input3_14 =  data['graph'][6][7]
            input3_15 =  data['graph'][6][8]
            input3_16 =  data['graph'][6][9]
            input3_17 =  data['graph'][6][10]
            input3_18 =  data['graph'][6][11]

            input3_19 =  data['graph'][7][0]
            input3_20 =  data['graph'][7][1]
            input3_21 =  data['graph'][7][2]
            input3_22 =  data['graph'][7][3]
            input3_23 =  data['graph'][7][4]
            input3_24 =  data['graph'][7][5]
            input3_25 =  data['graph'][7][6]
            input3_26 =  data['graph'][7][7]
            input3_27 =  data['graph'][7][8]
            input3_28 =  data['graph'][7][9]
            input3_29 =  data['graph'][7][10]
            input3_30 =  data['graph'][7][11]

            input3_31 =  data['graph'][8][0]
            input3_32 =  data['graph'][8][1]

            input3_33 =  data['graph'][9][0]
            input3_34 =  data['graph'][9][1]

            # ort_inputs = {ort_session.get_inputs()[0].name: input1.cpu().numpy(),ort_session.get_inputs()[1].name: input2.cpu().numpy(),ort_session.get_inputs()[2].name: input4.cpu().numpy(),ort_session.get_inputs()[3].name: input5.cpu().numpy(),ort_session.get_inputs()[4].name: input3_2.cpu().numpy(),ort_session.get_inputs()[5].name: input3_3.cpu().numpy(),ort_session.get_inputs()[6].name: input3_4.cpu().numpy(),ort_session.get_inputs()[7].name: input3_5.cpu().numpy(),ort_session.get_inputs()[8].name: input3_6.cpu().numpy(),ort_session.get_inputs()[9].name: input3_7.cpu().numpy(),ort_session.get_inputs()[10].name: input3_8.cpu().numpy(),ort_session.get_inputs()[11].name: input3_9.cpu().numpy(),ort_session.get_inputs()[12].name: input3_10.cpu().numpy(),ort_session.get_inputs()[13].name: input3_11.cpu().numpy(),ort_session.get_inputs()[14].name: input3_12.cpu().numpy(),ort_session.get_inputs()[15].name: input3_13.cpu().numpy(),ort_session.get_inputs()[16].name: input3_14.cpu().numpy(),ort_session.get_inputs()[17].name: input3_15.cpu().numpy(),ort_session.get_inputs()[18].name: input3_16.cpu().numpy(),ort_session.get_inputs()[19].name: input3_17.cpu().numpy(),ort_session.get_inputs()[20].name: input3_18.cpu().numpy(),ort_session.get_inputs()[21].name: input3_19.cpu().numpy(),ort_session.get_inputs()[22].name: input3_20.cpu().numpy(),ort_session.get_inputs()[23].name: input3_21.cpu().numpy(),ort_session.get_inputs()[24].name: input3_22.cpu().numpy(),ort_session.get_inputs()[25].name: input3_23.cpu().numpy(),ort_session.get_inputs()[26].name: input3_24.cpu().numpy(),ort_session.get_inputs()[27].name: input3_25.cpu().numpy(),ort_session.get_inputs()[28].name: input3_26.cpu().numpy(),ort_session.get_inputs()[29].name: input3_27.cpu().numpy(),ort_session.get_inputs()[30].name: input3_28.cpu().numpy(),ort_session.get_inputs()[31].name: input3_29.cpu().numpy(),ort_session.get_inputs()[32].name: input3_30.cpu().numpy(),ort_session.get_inputs()[33].name: input3_31.cpu().numpy(),ort_session.get_inputs()[34].name: input3_32.cpu().numpy(),ort_session.get_inputs()[35].name: input3_33.cpu().numpy(),ort_session.get_inputs()[36].name: input3_34.cpu().numpy()}
            ort_inputs = {
                        ort_session.get_inputs()[0].name: input1.cpu().numpy(),
                        ort_session.get_inputs()[1].name: input2.cpu().numpy(),
                        ort_session.get_inputs()[2].name: input4.cpu().numpy(),
                        ort_session.get_inputs()[3].name: input5.cpu().numpy(),
                        ort_session.get_inputs()[4].name: input3_2.cpu().numpy(),
                        ort_session.get_inputs()[5].name: input3_3.cpu().numpy(),
                        ort_session.get_inputs()[6].name: input3_4.cpu().numpy(),
                        ort_session.get_inputs()[7].name: input3_5.cpu().numpy(),
                        ort_session.get_inputs()[8].name: input3_6.cpu().numpy(),
                        ort_session.get_inputs()[9].name: input3_7.cpu().numpy(),
                        ort_session.get_inputs()[10].name: input3_8.cpu().numpy(),
                        ort_session.get_inputs()[11].name: input3_9.cpu().numpy(),
                        ort_session.get_inputs()[12].name: input3_10.cpu().numpy(),
                        ort_session.get_inputs()[13].name: input3_11.cpu().numpy(),
                        ort_session.get_inputs()[14].name: input3_12.cpu().numpy(),
                        ort_session.get_inputs()[15].name: input3_13.cpu().numpy(),
                        ort_session.get_inputs()[16].name: input3_14.cpu().numpy(),
                        ort_session.get_inputs()[17].name: input3_15.cpu().numpy(),
                        ort_session.get_inputs()[18].name: input3_16.cpu().numpy(),
                        ort_session.get_inputs()[19].name: input3_17.cpu().numpy(),
                        ort_session.get_inputs()[20].name: input3_18.cpu().numpy(),
                        ort_session.get_inputs()[21].name: input3_19.cpu().numpy(),
                        ort_session.get_inputs()[22].name: input3_20.cpu().numpy(),
                        ort_session.get_inputs()[23].name: input3_21.cpu().numpy(),
                        ort_session.get_inputs()[24].name: input3_22.cpu().numpy(),
                        ort_session.get_inputs()[25].name: input3_23.cpu().numpy(),
                        ort_session.get_inputs()[26].name: input3_24.cpu().numpy(),
                        ort_session.get_inputs()[27].name: input3_25.cpu().numpy(),
                        ort_session.get_inputs()[28].name: input3_26.cpu().numpy(),
                        ort_session.get_inputs()[29].name: input3_27.cpu().numpy(),
                        ort_session.get_inputs()[30].name: input3_28.cpu().numpy(),
                        ort_session.get_inputs()[31].name: input3_29.cpu().numpy(),
                        ort_session.get_inputs()[32].name: input3_30.cpu().numpy(),
                        ort_session.get_inputs()[33].name: input3_31.cpu().numpy(),
                        ort_session.get_inputs()[34].name: input3_32.cpu().numpy(),
                        ort_session.get_inputs()[35].name: input3_33.cpu().numpy(),
                        ort_session.get_inputs()[36].name: input3_34.cpu().numpy()
                        }
            ort_outs = ort_session.run(None, ort_inputs)

                # output = net(new_data)
                # output = script_mod(new_data)
                # print(output)
                #results = [x[0:1].detach().cpu().numpy() for x in output[1]]
            # results = ort_outs[1]
            # ort_outs_0 = torch.from_numpy(ort_outs[0])
            # ort_outs_1 = torch.from_numpy(ort_outs[1])
            results = [torch.from_numpy(ort_outs[1])]
            # output = output
            # print(output.shape)
            # for i in range(output[0].shape[0]):
            #     print("lrh", i)
            # output = net(new_data)
            # print(output[0])
            # print("------------------")
            # print(ort_outs[0][0])
            # np.testing.assert_allclose(to_numpy(output[0]), ort_outs[0][0], rtol=1e-03, atol=1e-05)
            # return
            # if i == 116:
            #     print(output[116])
            #     print(ort_outs[0][116])

            
        results = [x[0:1].detach().cpu().numpy() for x in results]
        # return
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
