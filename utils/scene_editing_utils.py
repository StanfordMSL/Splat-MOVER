from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
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

from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
from sklearn.cluster import HDBSCAN

from numpy.typing import NDArray
from nerfstudio.cameras.camera_utils import quaternion_slerp

# import pytorch3d as p3d
import roma

import itertools

from utils.nerf_utils import NeRF

# # # # #
# # # # # Utils
# # # # #


def in_convex_hull(points, convex_hull):
    return Delaunay(convex_hull).find_simplex(points) >= 0


def spherical_filter(
    source_points, target_points, radius=0.1, use_Mahalanobis_distance: bool = True
):
    if use_Mahalanobis_distance:
        # using a BallTree
        ball_tree = BallTree(
            source_points,
            metric="mahalanobis",
            V=np.diag(np.concatenate((np.repeat([1], 3), np.repeat([50], 3), [300]))),
        )

        # find the points within a sphere at each target point
        groups = ball_tree.query_radius(target_points, radius)

        # indices
        inds = np.unique(list(itertools.chain.from_iterable(groups)))

        return inds
    else:
        # using a KDTree
        kd_tree = cKDTree(source_points[:, :3])

        # find the points within a sphere at each target point
        groups = kd_tree.query_ball_point(target_points[:, :3], radius)

        # indices
        inds = np.unique(list(itertools.chain.from_iterable(groups)))

        return inds


def get_centroid(
    nerf: NeRF,
    env_pcd,
    pcd_attr: Dict,
    positives: str,
    negatives: str = "object, things, stuff, texture",
    threshold: float = 0.85,
    visualize_pcd: bool = True,
    enable_convex_hull: bool = False,
    enable_spherical_filter: bool = False,
    enable_clustering: bool = False,
    print_debug_info: bool = False,
    filter_radius: float = 0.05,
    obj_priors: Dict = {},
    use_Mahalanobis_distance: bool = True,
):

    # get the semantic outputs
    semantic_info = nerf.get_semantic_point_cloud(
        positives=positives, negatives=negatives, pcd_attr=pcd_attr
    )

    # initial point cloud for the scene
    scene_pcd = env_pcd

    # points in the point cloud
    scene_pcd_pts = np.asarray(scene_pcd.points)
    scene_pcd_colors = np.asarray(scene_pcd.colors)

    # threshold for masking the point cloud
    threshold_mask = threshold

    # scaled similarity
    sc_sim = torch.clip(semantic_info["similarity"] - 0.5, 0, 1)
    sc_sim = sc_sim / (sc_sim.max() + 1e-6)

    # depth mask
    depth_mask = None

    if len(scene_pcd.points) != semantic_info["similarity"].shape[0]:
        raise ValueError(
            "The Cosine similarity should be computed for all points in the point cloud!"
        )

    similarity_mask = (
        (sc_sim > threshold_mask)
        .squeeze()
        .reshape(
            -1,
        )
        .cpu()
        .numpy()
    )

    # masked point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(scene_pcd.points)[similarity_mask, ...]
    )
    pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(scene_pcd.colors)[similarity_mask, ...]
    )

    if visualize_pcd:
        fig = o3d.visualization.draw_plotly([pcd])
        fig.show()

    # # # # #
    # # # # # Outlier Removal
    # # # # #

    # threshold based on the standard deviation of average distances
    std_ratio = 0.01

    # remove outliers
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
    # alternative:
    pcd, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.05)

    if visualize_pcd:
        fig = o3d.visualization.draw_plotly([pcd])
        fig.show()

    # Point Cloud
    pcd_pts = np.asarray(pcd.points)

    # update the similarity mask
    similarity_mask_subset = np.zeros_like(
        similarity_mask[similarity_mask == True], dtype=bool
    )
    similarity_mask_subset[ind] = True

    similarity_mask_out = similarity_mask
    similarity_mask_out[similarity_mask == True] = similarity_mask_subset

    if enable_spherical_filter:
        # apply a spherical filter
        rel_inds = spherical_filter(
            source_points=np.concatenate(
                (scene_pcd_pts, scene_pcd_colors, sc_sim.cpu().numpy()), axis=-1
            ),
            target_points=np.concatenate(
                (
                    scene_pcd_pts[similarity_mask_out],
                    scene_pcd_colors[similarity_mask_out],
                    sc_sim.cpu().numpy()[similarity_mask_out],
                ),
                axis=-1,
            ),
            radius=filter_radius,
            use_Mahalanobis_distance=use_Mahalanobis_distance,
        )
        rel_inds = np.array(list(rel_inds)).astype(int)

        # update the mask
        similarity_mask_out[rel_inds] = True

        if print_debug_info:
            print(f"Spherical Filter Before : {len(pcd_pts)}, After: {len(rel_inds)}")

    # update the point cloud
    pcd_pts = scene_pcd_pts[similarity_mask_out]

    if enable_convex_hull:
        # compute the convex hull
        convex_hull = ConvexHull(pcd_pts)

        # examine the convex hull
        convex_hull_mask = in_convex_hull(scene_pcd_pts, pcd_pts[convex_hull.vertices])

        if print_debug_info:
            print(
                f"Convex Hull Proc. Before : {len(pcd_pts)}, After: {len(convex_hull_mask.nonzero()[0])}"
            )
    else:
        convex_hull_mask = np.zeros(len(scene_pcd_pts), dtype=bool)

    # update the similarity mask
    similarity_mask_out = np.logical_or(similarity_mask_out, convex_hull_mask)

    # TODO: Remove
    if visualize_pcd:
        # point cloud
        pcd_pts = scene_pcd_pts[similarity_mask_out]
        color_masked = np.asarray(scene_pcd.colors)[similarity_mask_out, ...]

        if visualize_pcd:
            print(len(pcd_pts))

        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)
        pcd.colors = o3d.utility.Vector3dVector(color_masked)

        fig = o3d.visualization.draw_plotly([pcd])
        fig.show()

    # apply prior information
    if obj_priors != {}:
        if "mask_prior" in obj_priors.keys():
            # compute the softmax
            probs = torch.nn.functional.softmax(
                torch.cat(
                    (obj_priors["mask_prior"], semantic_info["similarity"]), dim=-1
                ),
                dim=-1,
            )

            # maximizer
            pb_argmax = torch.argmax(probs, dim=-1, keepdim=True)

            # update the similarity mask
            similarity_mask_out = np.logical_and(
                similarity_mask_out, (pb_argmax == 1).squeeze().cpu().numpy()
            )

    # TODO: Remove
    if visualize_pcd:
        # point cloud
        pcd_pts = scene_pcd_pts[similarity_mask_out]
        color_masked = np.asarray(scene_pcd.colors)[similarity_mask_out, ...]

        if visualize_pcd:
            print(len(pcd_pts))

        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)
        pcd.colors = o3d.utility.Vector3dVector(color_masked)

        fig = o3d.visualization.draw_plotly([pcd])
        fig.show()

    # Clustering
    if enable_clustering:
        # clustering
        hdb_scan = HDBSCAN(
            min_cluster_size=5,
            metric="mahalanobis",
            metric_params={
                "V": np.diag(
                    np.concatenate((np.repeat([1], 3), np.repeat([5], 3), [10]))
                )
            },
        )

        # fit the data
        hdb_scan.fit(
            np.concatenate(
                (
                    scene_pcd_pts[similarity_mask_out],
                    np.asarray(scene_pcd.colors)[similarity_mask_out, ...],
                    sc_sim[similarity_mask_out, ...].cpu().numpy(),
                ),
                axis=-1,
            )
        )

        # all labels
        point_to_labels = hdb_scan.labels_.astype(int)
        unique_labels = set(hdb_scan.labels_)

        if print_debug_info:
            print(f"Unique Labels: {unique_labels}")

        # assigned colors
        colors = np.array(
            [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        )

        # visualize the clusters
        vis_pcd_pts = scene_pcd_pts[similarity_mask_out]
        vis_colors = colors[point_to_labels][:, :3]

        if visualize_pcd:
            # point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vis_pcd_pts)
            pcd.colors = o3d.utility.Vector3dVector(vis_colors)

            fig = o3d.visualization.draw_plotly([pcd])
            fig.show()

        # Select the cluster containing the point with the maximum similarity measure.

        # index of the maximizer of the similarity metric
        max_sim_idx = torch.argmax(sc_sim[similarity_mask_out])

        # Get the cluster containing the most similar point
        sel_cluster = point_to_labels[max_sim_idx]

        if visualize_pcd:
            vis_pcd_pts = scene_pcd_pts[similarity_mask_out][
                point_to_labels == sel_cluster
            ]
            vis_colors = color_masked[point_to_labels == sel_cluster]

            # point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vis_pcd_pts)
            pcd.colors = o3d.utility.Vector3dVector(vis_colors)

            fig = o3d.visualization.draw_plotly([pcd])
            fig.show()

        # mask from the clustering procedure
        clustering_mask = point_to_labels == sel_cluster

        # update the similarity mask
        similarity_mask_out[similarity_mask_out == True] = np.logical_and(
            similarity_mask_out[similarity_mask_out == True], clustering_mask
        )

        # points in the point cloud
        pts_cond = np.asarray(pcd.points)

        if enable_convex_hull:
            # compute the convex hull
            convex_hull = ConvexHull(pts_cond)

            # examine the convex hull
            convex_hull_mask = in_convex_hull(
                scene_pcd_pts, pts_cond[convex_hull.vertices]
            )

            if print_debug_info:
                print(
                    f"Convex Hull Proc. Before : {len(pts_cond)}, After: {len(convex_hull_mask.nonzero()[0])}"
                )

        # update the similarity mask
        similarity_mask_out = np.logical_or(similarity_mask_out, convex_hull_mask)

        if visualize_pcd:
            vis_pcd_pts = scene_pcd_pts[similarity_mask_out]
            vis_colors = np.asarray(scene_pcd.colors)[similarity_mask_out, ...]

            # point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vis_pcd_pts)
            pcd.colors = o3d.utility.Vector3dVector(vis_colors)

            fig = o3d.visualization.draw_plotly([pcd])
            fig.show()

    # compute the desired centroid
    centroid = np.mean(pcd_pts, axis=0)

    if visualize_pcd:
        # point cloud
        pcd_pts = scene_pcd_pts[similarity_mask_out]
        color_masked = np.asarray(scene_pcd.colors)[similarity_mask_out, ...]

        if print_debug_info:
            print(len(pcd_pts))

        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)
        pcd.colors = o3d.utility.Vector3dVector(color_masked)

        fig = o3d.visualization.draw_plotly([pcd])
        fig.show()

    # compute the bounds in the z-direction
    z_bounds = (np.amin(pcd_pts[:, -1]), np.amax(pcd_pts[:, -1]))

    # other attributes
    obj_attributes = {
        "convex_hull": convex_hull_mask if enable_convex_hull else [],
        "spherical_filter": rel_inds if enable_spherical_filter else [],
        "clustering_mask": clustering_mask if enable_clustering else [],
        "raw_similarity": semantic_info["raw_similarity"],
    }

    return centroid, z_bounds, scene_pcd, similarity_mask_out, obj_attributes


def get_interpolated_gaussians(
    means_a: NDArray,
    means_b: NDArray,
    quats_a: NDArray,
    quats_b: NDArray,
    des_rot: NDArray,
    steps: int = 10,
    tangent: NDArray = torch.tensor([[0.1, 0.1, 0]], device="cuda"),
    demo_height: float = 0.025,
    rot_origin: NDArray = None,
) -> List[float]:
    """Return interpolation of poses with specified number of steps.
    Args:
        means_a: initial means
        means_b: final means
        quats_a: initial quaternions
        quats_b: final quaternions
        des_rot: desired body-frame rotation
        steps: number of steps the interpolated path should contain
        demo_height: desired offset for the translation component (for demos)
        rot_origin: origin of the rotation axis
    """

    # device
    device = means_a.device

    # normalize the quaternions
    quats_a = torch.nn.functional.normalize(quats_a, dim=-1)
    quats_b = torch.nn.functional.normalize(quats_b, dim=-1)

    # t_steps
    ts_a = torch.linspace(0, 1, steps // 3).to(device)[:, None, None]

    ts_a_b = torch.linspace(0, 1, steps // 3).to(device)[:, None, None]

    ts_b = torch.linspace(0, 1, steps - len(ts_a) - len(ts_a_b)).to(device)[
        :, None, None
    ]

    center_param = 0.33
    center = (1 - center_param) * means_a + center_param * means_b
    center[:, -1] += demo_height

    # interpolated means
    interpolated_means_a = (1 - ts_a) * means_a[None] + ts_a * center[None]

    center_param = 0.9
    center_a_b = (1 - center_param) * means_a + center_param * means_b
    center_a_b[:, -1] = center[:, -1]

    # interpolated means
    interpolated_means_a_b = (1 - ts_a_b) * interpolated_means_a[
        -1:, ...
    ] + ts_a_b * center_a_b[None]

    # interpolated means
    interpolated_means_b = (1 - ts_b) * interpolated_means_a_b[
        -1:, ...
    ] + ts_b * means_b[None]

    interpolated_means = torch.cat(
        (interpolated_means_a, interpolated_means_a_b, interpolated_means_b), dim=0
    )

    # interpolated rotations
    interpolated_rot = roma.utils.unitquat_slerp(
        torch.tensor([0, 0, 0, 1.0], device=device)[None],
        des_rot,
        torch.linspace(0, 1, steps).to(device),
    )

    # center of mass of the Gaussians
    com_means = torch.mean(interpolated_means, dim=1, keepdim=True)
    if rot_origin is None:
        rot_origin = com_means
    else:
        rot_origin = rot_origin.expand(*com_means.shape)

    # rigid transformation
    rigid_trans = roma.RigidUnitQuat(
        linear=interpolated_rot, translation=torch.zeros_like(interpolated_rot)[..., :3]
    )

    # rotate the Gaussians
    transformed_means = rigid_trans.apply(interpolated_means - rot_origin)
    interpolated_means = transformed_means + rot_origin

    # interpolate quaternions
    interpolated_quats = roma.quat_xyzw_to_wxyz(
        roma.quat_product(interpolated_rot, roma.quat_wxyz_to_xyzw(quats_a[None]))
    )

    return interpolated_means, interpolated_quats


def get_bezier_interpolation(
    means_a: NDArray,
    means_b: NDArray,
    steps: int = 10,
    tangent: NDArray = torch.tensor([[0.1, 0.1, 0]], device="cuda"),
) -> List[float]:
    """Return interpolation of poses with specified number of steps using a Bézier Curve.
    Args:
        means_a: initial means
        means_b: final means
        steps: number of steps the interpolated path should contain
    """
    # device
    device = means_a.device

    # timesteps
    t_steps = torch.linspace(0.0, 1.0, steps).to(device)

    # time matrix
    T_mat = torch.stack(
        (t_steps**3.0, t_steps**2.0, t_steps, torch.ones_like(t_steps)), dim=-1
    )

    # Basis matrix
    M_b_mat = torch.tensor(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]], device=device
    ).float()

    # Time-and-Basis matrix
    T_M_b_mat = (T_mat @ M_b_mat)[:, None, None, :]

    # Geometry Matrix
    G_mat = means_a.clone()
    G_mat = torch.stack((G_mat, means_a + tangent, means_b - tangent, means_b), dim=1)[
        None
    ]

    # interpolated points
    return (T_M_b_mat @ G_mat).squeeze()
