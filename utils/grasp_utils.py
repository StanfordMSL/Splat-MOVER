# # # # #
# # # # # Grasp Generation
# # # # #

import argparse

from graspnet.graspnetAPI.graspnetAPI.grasp import GraspGroup
from graspnet.graspnet_baseline.models.graspnet import GraspNet, pred_decode
from graspnet.graspnet_baseline.utils.collision_detector import (
    ModelFreeCollisionDetector,
)
import scipy.io as scio
from PIL import Image
import open3d as o3d
from pathlib import Path
import numpy as np
import torch
import gc


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class Args(argparse.Namespace):
    # Model checkpoint path
    checkpoint_path = Path("")

    # Point Number [default: 20000]
    num_point = 20000

    # View Number [default: 300]
    num_view = 300

    # Collision Threshold in collision detection [default: 0.01]
    collision_thresh = 0.01

    # Voxel Size to process point clouds before collision detection [default: 0.01
    voxel_size = 0.01


def get_affordance(tree, points, env_affordance, n_neighbors=1):
    """
    Computes the affordance for each point in points in the environment.
    """
    # nearest-neighbors
    dist, ind = tree.query(points, k=n_neighbors)

    # alternative
    # ind = tree.query_ball_point(points, r=k_tree_rad)

    # compute affordance
    afford = env_affordance[ind].detach().cpu().numpy()

    return afford


def rank_grasps(grasps, env_tree, env_affordance, n_neighbors=1):
    """
    Utilizes the composite grasp-affordance metric to rank candidate grasps.
    """
    # grasp points
    grasp_pts = []

    # grasp score from GraspNet
    grasp_score = []

    for gp in grasps:
        # grasp points
        grasp_pts.append(gp.translation)

        # grasp score from grasp net
        grasp_score.append(gp.score)

    grasp_pts = np.array(grasp_pts)
    grasp_score = np.array(grasp_score)

    # compute the affordance
    grasp_affordance = get_affordance(
        env_tree, grasp_pts, env_affordance, n_neighbors=n_neighbors
    )

    # compute the composite metric
    comp_score = grasp_affordance.squeeze()

    # Alternative:
    # comp_score = sigmoid(grasp_score) * grasp_affordance.squeeze()

    # ordering
    rank_order = np.argsort(comp_score)[::-1]

    # rerank grasps
    return grasps[rank_order], comp_score[rank_order]


def get_net(cfgs):
    # Init the model
    net = GraspNet(
        input_feature_dim=0,
        num_view=cfgs.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"]
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net


def get_and_process_data(cloud_masked, color_masked, cfgs):
    # upsample or downsample the point cloud
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(
            len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True
        )
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device).float()
    end_points["point_clouds"] = cloud_sampled
    end_points["cloud_colors"] = color_sampled

    # point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_masked)
    pcd.colors = o3d.utility.Vector3dVector(color_masked)

    return end_points, pcd


def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud, cfgs):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(
        gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh
    )
    gg = gg[~collision_mask]
    return gg


def vis_grasps(gg, cloud, num_vis_grasp, grasp_group_color=np.array([[0, 1, 0]])):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:num_vis_grasp]
    grippers = gg.to_open3d_geometry_list()
    # refine visualization
    grippers_pcd = []
    for gp in grippers:
        gp.paint_uniform_color(grasp_group_color.T)
        grippers_pcd.append(gp.sample_points_uniformly(number_of_points=4000))
    # o3d.visualization.draw_geometries([cloud, *grippers])
    o3d.visualization.draw_plotly([cloud, *grippers, *grippers_pcd])


def demo(
    cloud_masked,
    color_masked,
    cfgs,
    num_vis_grasp=50,
    grasp_group_color=np.array([[0, 1, 0]]),
):
    net = get_net(cfgs)
    end_points, cloud = get_and_process_data(cloud_masked, color_masked, cfgs)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points), cfgs)
    # vis_grasps(gg, cloud, num_vis_grasp, grasp_group_color)

    del net
    gc.collect()
    torch.cuda.empty_cache()

    return gg, cloud


def visualize_grasps(
    grasps,
    pcd,
    num_vis_grasp=50,
    grasp_group_color=np.array([[0, 1, 0]]),
    showaxes_grid=True,
    width=600,
    height=600,
):
    """
    Visualize grasps.
    """
    grippers = grasps[:num_vis_grasp].to_open3d_geometry_list()

    # refine visualization
    grippers_pcd = []
    for gp in grippers:
        gp.paint_uniform_color(grasp_group_color.T)
        grippers_pcd.append(gp.sample_points_uniformly(number_of_points=4000))
    # o3d.visualization.draw_geometries([cloud, *grippers])
    return o3d.visualization.draw_plotly(
        [pcd, *grippers, *grippers_pcd],
        showaxes_grid=showaxes_grid,
        width=width,
        height=height,
    )


def save_grasps(
    filename: Path, grasps, translation: np.ndarray = None, num_grasps_save: int = 10
):
    # create the directory, if necessary
    filename.parent.mkdir(parents=True, exist_ok=True)

    # get transformation matrix
    grasp_pose = []

    for idx in range(num_grasps_save):
        # grasp pose
        gp_T = np.eye(4)

        # rotation
        gp_T[:3, :3] = grasps[idx].rotation_matrix

        # translation
        gp_T[:3, -1] = grasps[idx].translation

        # append
        grasp_pose.append(gp_T)

    # grasp pose
    grasp_pose = np.array(grasp_pose)

    # save file
    np.save(file=filename, arr=grasp_pose)

    # save file
    if translation is not None:
        np.save(file=f"{filename.stem}_translation.npy", arr=translation)


def reorient_grasps(
    grasps, normal_dir, ang_threshold=np.deg2rad(20), side_grasp_desired: bool = False
):
    import copy
    from scipy.spatial.transform import Rotation

    # modified grasp poses
    mod_grasps = copy.deepcopy(grasps)

    # rotation matrices
    rot_mat = mod_grasps.rotation_matrices

    # x-axis of each grasp
    x_axis = rot_mat[..., 0]

    # z-axis of each grasp
    z_axis = rot_mat[..., 2]

    if not side_grasp_desired:
        # align the x-axis with the negative normal direction

        # negative normal direction
        normal_dir = -np.reshape(normal_dir, (-1, 1))

        # angle between the normal direction and the x-axis of the grasp pose
        ang_x_to_normal = np.arccos(
            (x_axis @ normal_dir)
            / (
                np.linalg.norm(x_axis, axis=-1, keepdims=True)
                * np.linalg.norm(normal_dir)
            )
        )

        # rotation axis
        rot_ax = np.cross(x_axis, normal_dir.reshape(1, -1), axis=-1)

        # normalize
        rot_ax /= np.linalg.norm(rot_ax, axis=-1, keepdims=True)

        # rotate the grasp about the rotation axis of the grasp pose
        rot_obj = Rotation.from_rotvec(ang_x_to_normal * rot_ax)

        # result of the rotation
        prop_rot = rot_obj.as_matrix() @ rot_mat
        mod_grasps.rotation_matrices = prop_rot
    else:
        # align the z-axis with the positive normal direction

        # negative normal direction
        normal_dir = np.reshape(normal_dir, (-1, 1))

        # angle between the normal direction and the z-axis of the grasp pose
        ang_z_to_normal = np.arccos(
            (z_axis @ normal_dir)
            / (
                np.linalg.norm(z_axis, axis=-1, keepdims=True)
                * np.linalg.norm(normal_dir)
            )
        )

        # rotation axis
        rot_ax = np.cross(z_axis, normal_dir.reshape(1, -1), axis=-1)

        # normalize
        rot_ax /= np.linalg.norm(rot_ax, axis=-1, keepdims=True)

        # rotate the grasp about the rotation axis of the grasp pose
        rot_obj = Rotation.from_rotvec(ang_z_to_normal * rot_ax)

        # result of the rotation
        prop_rot = rot_obj.as_matrix() @ rot_mat
        mod_grasps.rotation_matrices = prop_rot

    return mod_grasps
