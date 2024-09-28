# %%
from __future__ import annotations

import json
import os, sys
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from typing_extensions import Annotated
import pickle
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib as mpl
from tqdm import tqdm
import open3d as o3d
from scipy.spatial import KDTree

# modify Python's Path
sys.path.insert(1, f"{Path(__file__).parent.parent.resolve()}")

from utils.nerf_utils import *
from utils.grasp_utils import *

# %%
# # # # #
# # # # # Config Path
# # # # #

# # mode
gaussian_splatting = True

if gaussian_splatting:
    # ARMLAB:
    config_path = Path(f"<config.yml>")
else:
    # ASK-NeRF
    config_path = Path(f"<config.yml>")
# %%
# rescale factor
res_factor = None

# option to enable visualization of the environment point cloud
enable_visualization_pcd = False

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize NeRF
nerf = NeRF(
    config_path=config_path,
    res_factor=res_factor,
    test_mode="test",  # Options: "inference", "val", "test"
    dataset_mode="train",
    device=device,
)

# camera intrinsics
H, W, K = nerf.get_camera_intrinsics()
K = K.to(device)

# poses in test dataset
eval_poses = nerf.get_poses()

# images for evaluation
eval_imgs = nerf.get_images()

# option to generate a dense point cloud
generate_dense_pcd = False

# option to generate a mesh
generate_mesh = False

if generate_dense_pcd:
    from utils.exporter_utils import generate_point_cloud

    # point cloud attributes
    env_attr = {}

    # generate point cloud
    env_pcd, env_attr["clip_embeds"] = generate_point_cloud(
        pipeline=nerf.pipeline,
        num_points=1000000,
        remove_outliers=True,
        estimate_normals=False,
        reorient_normals=False,
        rgb_output_name="rgb",
        depth_output_name="depth",
        normal_output_name=None,
        use_bounding_box=True,
        bounding_box_min=(-1, -1, -0.6),
        bounding_box_max=(1, 1, 0.2),
        crop_obb=None,
        std_ratio=0.2,
    )

    if enable_visualization_pcd:
        # downsample the point cloud
        pcd_down = env_attr.voxel_down_sample(0.01)

        # visualize point cloud
        o3d.visualization.draw_plotly([pcd_down])
else:
    # generate the point cloud of the environment
    env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(use_bounding_box=False)

    if enable_visualization_pcd:
        # visualize point cloud
        o3d.visualization.draw_plotly([env_pcd])

# generate a mesh
if generate_mesh:
    from utils.exporter import ExportPoissonMesh

    # generate Poisson Mesh
    mesh = ExportPoissonMesh(
        load_config=config_path,
        output_dir=Path("renders/mesh"),
        normal_method="open3d",
        bounding_box_min=(-1, -1, -0.6),
        bounding_box_max=(1, 1, 0.2),
        texture_method="point_cloud",
        std_ratio=0.2,
    )

    mesh.main()

# %%
# # # # #
# # # # # Semantic Query
# # # # #

# list of positives
positives = "orange and black powerdrill"

# update list of negatives ['things', 'stuff', 'object', 'texture']: 'object, things, stuff, texture'
negatives = "object, things, stuff, texture"

# option to render the point cloud of the entire environment or from a camera
camera_semantic_pcd = False

if camera_semantic_pcd:
    # camera pose
    cam_pose = eval_poses[50]  # 11 (drill); 9 (hammer), 13 (mug)

    # generate semantic RGB-D point cloud
    cam_rgb, cam_pcd_points, grasp_pcd, depth_mask, cam_out = (
        nerf.generate_RGBD_point_cloud(
            pose=cam_pose,
            save_image=True,
            filename="figures/eval_cam_rgb.png",
            compute_semantics=True,
            positives=positives,
            negatives=negatives,
        )
    )

    # semantic outputs for grasp generation
    semantic_info = cam_out
else:
    # get the semantic outputs
    semantic_info = nerf.get_semantic_point_cloud(
        positives=positives, negatives=negatives, pcd_attr=env_attr
    )

    # initial point cloud for grasp generation
    grasp_pcd = env_pcd

# threshold for masking the point cloud
threshold_mask = 0.85

# scaled similarity
sc_sim = torch.clip(semantic_info["similarity"] - 0.5, 0, 1)
sc_sim = sc_sim / (sc_sim.max() + 1e-6)

if not camera_semantic_pcd:
    depth_mask = None

    if len(grasp_pcd.points) != semantic_info["similarity"].shape[0]:
        raise ValueError(
            "The Cosine similarity should be computed for all points in the point cloud!"
        )

if depth_mask is not None:
    similarity_mask = (
        (sc_sim > threshold_mask)[depth_mask]
        .squeeze()
        .reshape(
            -1,
        )
        .cpu()
        .numpy()
    )
else:
    similarity_mask = (
        (sc_sim > threshold_mask)
        .squeeze()
        .reshape(
            -1,
        )
        .cpu()
        .numpy()
    )

# masked point cloud for grasp generation
cloud_masked = np.asarray(grasp_pcd.points)[similarity_mask, ...]
color_masked = np.asarray(grasp_pcd.colors)[similarity_mask, ...]

# option to enable visualization of the masked point cloud
enable_visualization_masked_pcd = True

# create the point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud_masked)
pcd.colors = o3d.utility.Vector3dVector(color_masked)

if enable_visualization_masked_pcd:
    o3d.visualization.draw_plotly([pcd])

# # # # #
# # # # # Outlier Removal
# # # # #

# threshold based on the standard deviation of average distances
std_ratio = 0.1

# remove outliers
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)

# alternative
# pcd, ind = pcd.remove_radius_outlier(nb_points=50, radius=0.05)

if enable_visualization_masked_pcd:
    o3d.visualization.draw_plotly([pcd])
# %%
# # # # #
# # # # # Grasp Generation
# # # # #

# configuration
cfgs = Args()

# checkpoint path
cfgs.checkpoint_path = Path(
    f"{Path(__file__).parent.parent.resolve()}/graspnet/model_checkpoints/checkpoint-rs.tar"
)

# number of grasps to visualize
num_vis_grasp = 10  # default: 50

# color to display each grasp
grasp_group_color = np.array([[0, 1, 0]])

# point cloud and endpoints
# sample points
cloud_masked = np.asarray(pcd.points)
color_masked = np.asarray(pcd.colors)

# # # # #
# # # # # Proposed Grasps from GraspNet
# # # # #

# proposed grasps from GraspNet
cand_grasps, pcd = demo(cloud_masked, color_masked, cfgs)

cand_grasps.nms()
cand_grasps.sort_by_score()
gg = cand_grasps

# visualize the grasps
visualize_grasps(
    gg, pcd=pcd, num_vis_grasp=num_vis_grasp, grasp_group_color=grasp_group_color
)

# # # # #
# # # # # Incorporate Affordance
# # # # #

# number of neighbors in query
n_gs_neighbors = 1

# radius for query
k_tree_rad = 5e-3

# construct KD-tree
env_tree = KDTree(data=np.asarray(env_pcd.points))

# affordance for the environment
env_affordance = nerf.pipeline.model.affordance

if env_pcd_mask is not None:
    # affordance for the environment
    env_affordance = env_affordance[env_pcd_mask]

# affordance-aware grasps
ranked_grasps, comp_score = rank_grasps(
    cand_grasps,
    env_tree=env_tree,
    env_affordance=env_affordance,
    n_neighbors=n_gs_neighbors,
)

# visualize the grasps
visualize_grasps(
    ranked_grasps,
    pcd=pcd,
    num_vis_grasp=num_vis_grasp,
    grasp_group_color=grasp_group_color,
)

# # # # #
# # # # # Save the Grasps.
# # # # #

# filename
pose_filename = Path(f"results/grasp_test.npy")

# number of grasps to save
num_grasps_save = 10

# save the proposed grasps
save_grasps(
    filename=pose_filename, grasps=ranked_grasps, num_grasps_save=num_grasps_save
)
# %%
