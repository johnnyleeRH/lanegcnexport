from collections import OrderedDict
import copy
import math
import os
import sys
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple, Union

from argoverse.map_representation.map_api import ArgoverseMap
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)



'''refer to argoverse-forcasting prj'''

'''
input_: np.ndarray (20, 2)
preds: np.ndarray (n, 30, 2)
gt: np.ndarray(30, 2) or none
city: str
'''

def viz_predictions(
        input_: np.ndarray,
        preds: np.ndarray,
        gt: np.ndarray,
        city: np.ndarray,
) -> None:
    num_tracks = 1
    # obs_len = input_.shape[1]
    # pred_len = target.shape[1]

    plt.figure(0, figsize=(8, 7))
    avm = ArgoverseMap()
    # input yellow
    for i in range(num_tracks):
        plt.plot(
            input_[:, 0],
            input_[:, 1],
            color="#ECA154",
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
        )
        plt.plot(
            input_[-1, 0],
            input_[-1, 1],
            "o",
            color="#ECA154",
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
            markersize=9,
        )
        if gt.any():
            # target red
            plt.plot(
                gt[:, 0],
                gt[:, 1],
                color="#d33e4c",
                label="Target",
                alpha=1,
                linewidth=3,
                zorder=15,
            )
            plt.plot(
                gt[-1, 0],
                gt[-1, 1],
                "o",
                color="#d33e4c",
                label="Target",
                alpha=1,
                linewidth=3,
                zorder=15,
                markersize=9,
            )
    '''
    for j in range(len(centerlines[i])):
        plt.plot(
            centerlines[i][j][:, 0],
            centerlines[i][j][:, 1],
            "--",
            color="grey",
            alpha=1,
            linewidth=1,
            zorder=0,
        )
    '''
    # predict green
    for j in range(len(preds)):
        plt.plot(
            preds[j][:, 0],
            preds[j][:, 1],
            color="#007672",
            label="Predicted",
            alpha=1,
            linewidth=1,
            zorder=20,
        )
        plt.plot(
            preds[j][-1, 0],
            preds[j][-1, 1],
            "o",
            color="#007672",
            label="Predicted",
            alpha=1,
            linewidth=1,
            zorder=20,
            markersize=8-j,
        )
        for k in range(len(preds[0])):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                preds[j][k, 0],
                preds[j][k, 1],
                city,
                query_search_range_manhattan=2.5,
            )
            [avm.draw_lane(lane_id, city) for lane_id in lane_ids]

    for j in range(len(input_)):
        lane_ids = avm.get_lane_ids_in_xy_bbox(
            input_[j, 0],
            input_[j, 1],
            city,
            query_search_range_manhattan=2.5,
        )
        [avm.draw_lane(lane_id, city) for lane_id in lane_ids]

    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.show(block=False)
    plt.pause(2)
    plt.close()