# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export utils such as structs, point cloud generation, and rendering code.
"""


from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pymeshlab
import torch
from jaxtyping import Float
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn

import open3d as o3d


def generate_point_cloud(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    reorient_normals: bool = False,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    use_bounding_box: bool = True,
    bounding_box_min: Optional[Tuple[float, float, float]] = None,
    bounding_box_max: Optional[Tuple[float, float, float]] = None,
    crop_obb: Optional[OrientedBox] = None,
    std_ratio: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        reorient_normals: Whether to re-orient the normals based on the view direction.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )
    points = []
    rgbs = []
    normals = []
    # CLIP embeddings
    clip_embeds = []

    if use_bounding_box and (crop_obb is not None and bounding_box_max is not None):
        CONSOLE.print(
            "Provided aabb and crop_obb at the same time, using only the obb",
            style="bold yellow",
        )
    with progress as progress_bar:
        # number of cameras
        num_cameras = len(pipeline.datamanager.train_dataset.cameras)

        # task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        task = progress_bar.add_task("Generating Point Cloud", total=num_cameras)
        while not progress_bar.finished:
            normal = None

            with torch.no_grad():
                camera, _ = pipeline.datamanager.next_train(0)
                assert isinstance(camera, Cameras)
                outputs = pipeline.model(camera)
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(
                    f"Could not find {rgb_output_name} in the model outputs",
                    justify="center",
                )
                CONSOLE.print(
                    f"Please set --rgb_output_name to one of: {outputs.keys()}",
                    justify="center",
                )
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(
                    f"Could not find {depth_output_name} in the model outputs",
                    justify="center",
                )
                CONSOLE.print(
                    f"Please set --depth_output_name to one of: {outputs.keys()}",
                    justify="center",
                )
                sys.exit(1)

            # get RGBA image
            rgba = outputs["rgb"]
            rgba = torch.cat((rgba, outputs["accumulation"]), dim=-1)

            depth = outputs[depth_output_name]

            # CLIP embeddings
            clip_embed = outputs["clip"]

            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(
                        f"Could not find {normal_output_name} in the model outputs",
                        justify="center",
                    )
                    CONSOLE.print(
                        f"Please set --normal_output_name to one of: {outputs.keys()}",
                        justify="center",
                    )
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0

            # project points into the world frame
            # camera intrinsics
            H, W, K = (
                camera.height.item(),
                camera.width.item(),
                camera.get_intrinsics_matrices(),
            )
            K = K.squeeze()

            # unnormalized pixel coordinates
            u_coords = torch.arange(W, device=rgba.device)
            v_coords = torch.arange(H, device=rgba.device)

            # meshgrid
            U_grid, V_grid = torch.meshgrid(u_coords, v_coords, indexing="xy")

            # transformed points in camera frame
            # [u, v, 1] = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]] @ [x/z, y/z, 1]
            cam_pts_x = (U_grid - K[0, 2]) * depth.squeeze() / K[0, 0]
            cam_pts_y = (V_grid - K[1, 2]) * depth.squeeze() / K[1, 1]

            cam_pcd_points = torch.stack(
                (cam_pts_x, cam_pts_y, depth.squeeze(), torch.ones_like(cam_pts_y)),
                axis=-1,
            )

            # camera pose
            cam_pose = torch.eye(4, device=rgba.device)
            cam_pose[:3] = camera.camera_to_worlds

            # convert from OpenGL to OpenCV Convention
            cam_pose[:, 1] = -cam_pose[:, 1]
            cam_pose[:, 2] = -cam_pose[:, 2]

            # point = torch.einsum('ij,hkj->hki', cam_pose, cam_pcd_points)

            point = cam_pose @ cam_pcd_points.view(-1, 4).T
            point = point.T.view(*cam_pcd_points.shape[:2], 4)
            point = point[..., :3]

            # Filter points with opacity lower than 0.5
            mask = rgba[..., -1] > 0.5

            point = point[mask]
            rgb = rgba[mask][..., :3]
            clip_embed = clip_embed[mask]

            if normal is not None:
                normal = normal[mask]

            if use_bounding_box:
                if crop_obb is None:
                    comp_l = torch.tensor(bounding_box_min, device=point.device)
                    comp_m = torch.tensor(bounding_box_max, device=point.device)
                    assert torch.all(
                        comp_l < comp_m
                    ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
                    mask = torch.all(
                        torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1
                    )
                else:
                    mask = crop_obb.within(point)
                point = point[mask]
                rgb = rgb[mask]
                clip_embed = clip_embed[mask]
                if normal is not None:
                    normal = normal[mask]

            points.append(point)
            rgbs.append(rgb)
            clip_embeds.append(clip_embed)
            if normal is not None:
                normals.append(normal)

            progress.advance(task, 1)

    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)
    clip_embeds = torch.cat(clip_embeds, dim=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

    ind = None
    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")
        if ind is not None:
            clip_embeds = clip_embeds[ind]

    # either estimate_normals or normal_output_name, not both
    if estimate_normals:
        if normal_output_name is not None:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(
                "Cannot estimate normals and use normal_output_name at the same time",
                justify="center",
            )
            sys.exit(1)
        CONSOLE.print("Estimating Point Cloud Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
    elif normal_output_name is not None:
        normals = torch.cat(normals, dim=0)
        if ind is not None:
            # mask out normals for points that were removed with remove_outliers
            normals = normals[ind]
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    return pcd, clip_embeds


def post_process_point_cloud(
    pcd: o3d.geometry.PointCloud,
    std_ratio: float = 10.0,
) -> o3d.geometry.PointCloud:

    # remove outliers
    CONSOLE.print("Cleaning Point Cloud")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
    print("\033[A\033[A")
    CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")

    # estimate normals
    CONSOLE.print("Estimating Point Cloud Normals")
    pcd.estimate_normals()
    print("\033[A\033[A")
    CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")

    return pcd
