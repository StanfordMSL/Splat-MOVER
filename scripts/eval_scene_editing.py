# %%
from __future__ import annotations

import json
import os, sys
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
from pytorch3d import transforms as py_transforms
from torch.distributions.multivariate_normal import MultivariateNormal
import open3d as o3d
from enum import Enum
from nerfstudio.utils.rich_utils import CONSOLE

# modify Python's Path
sys.path.insert(1, f"{Path(__file__).parent.parent.resolve()}")

from utils.nerf_utils import *
from utils.scene_editing_utils import *


# # # # #
# # # # # Config Path
# # # # #

# mode
gaussian_splatting = True

if gaussian_splatting:
    # config path
    config_path = Path(f"<config.yml>")
else:
    # ASK-NeRF
    config_path = Path(f"<config.yml>")

# name of the scene
scene_name = "scene_name"

# number of trials
num_trials = 10

# option to enable minimal visualization
enable_minimal_visualization = False

for trial_idx in range(num_trials):
    # # # # #
    # # # # # Load the Gaussian Splatting Model
    # # # # #

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
        dataset_mode="test",
        device=device,
    )

    # camera intrinsics
    H, W, K = nerf.get_camera_intrinsics()
    K = K.squeeze()
    K = K.to(device)

    # poses in test dataset
    eval_poses = nerf.get_poses()

    # images for evaluation
    eval_imgs = nerf.get_images()

    # option to generate a dense point cloud
    generate_dense_pcd = False

    # option to generate a mesh
    generate_dense_mesh = False

    # mesh-generation algorithm
    use_alpha_mesh = True

    # option to save the environment point cloud
    save_pcd = True

    # filename of the point cloud
    pcd_output_dir = Path("renders/pcd")

    # option to save the environment mesh
    save_mesh = True

    # filename of the point cloud
    mesh_output_dir = Path("renders/mesh")

    # option to save the point cloud at each stage of the manipulation task
    save_multi_stage_pcd = True

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
        env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(
            use_bounding_box=True
        )

        if enable_visualization_pcd:
            # visualize point cloud
            o3d.visualization.draw_plotly([env_pcd])

    # generate a mesh
    if generate_dense_mesh:
        from utils.exporter import ExportPoissonMesh

        # generate Poisson Mesh
        mesh = ExportPoissonMesh(
            load_config=config_path,
            output_dir=mesh_output_dir,
            normal_method="open3d",
            bounding_box_min=(-1, -1, -0.6),
            bounding_box_max=(1, 1, 0.2),
            texture_method="point_cloud",
            std_ratio=0.2,
            save_point_cloud=save_pcd,
            save_mesh=save_mesh,
        )

        mesh.main()

    # save the point cloud
    if trial_idx == 0:
        if save_pcd or (save_mesh and not generate_dense_mesh):
            from utils.exporter_utils import post_process_point_cloud

            # post-process the point cloud
            std_ratio = 0.2
            pcd = post_process_point_cloud(pcd=env_pcd, std_ratio=std_ratio)

            # create the directory, if necessary
            pcd_output_dir.mkdir(parents=True, exist_ok=True)

            # save the point cloud
            o3d.io.write_point_cloud(f"{pcd_output_dir}/point_cloud.ply", pcd)

            if save_mesh:
                # create the directory, if necessary
                mesh_output_dir.mkdir(parents=True, exist_ok=True)

                if use_alpha_mesh:
                    # alpha value
                    alpha_mesh = 0.01
                    mesh = (
                        o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                            pcd, alpha_mesh
                        )
                    )
                else:
                    # create a mesh from a sparse point cloud using the Poisson algorithm
                    CONSOLE.print("Computing Mesh... this may take a while.")
                    mesh, densities = (
                        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                            pcd, depth=9
                        )
                    )
                    vertices_to_remove = densities < np.quantile(densities, 0.1)
                    mesh.remove_vertices_by_mask(vertices_to_remove)
                    print("\033[A\033[A")
                    CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

                # save the mesh
                CONSOLE.print("Saving Mesh...")
                o3d.io.write_triangle_mesh(f"{mesh_output_dir}/poisson_mesh.ply", mesh)
                print("\033[A\033[A")
                CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

    # #  %%
    # # # # #
    # # # # # Scene Editing from the Gaussian Means: Objects to Move
    # # # # #

    # option to print debug info
    print_debug_info: bool = True

    # Set of all fruits
    fruits = {"fruit", "apple", "orange", "pear", "tomato"}

    # objects to move
    objects: List[str] = ["saucepan", "glass lid", "knife", "orange"]
    targets: List[str] = ["electric burner coil", "cutting board"]
    object_to_target: Dict[str, str] = {
        "saucepan": targets[0],
        "glass lid": targets[0],
        "knife": targets[1],
        "orange": targets[1],
    }

    # table location
    table_centroid, table_z_bounds, scene_pcd, table_sim_mask, table_attr = (
        get_centroid(
            nerf=nerf,
            env_pcd=env_pcd,
            pcd_attr=env_attr,
            positives="table",
            threshold=0.7,
            enable_convex_hull=True,
            enable_spherical_filter=False,
            visualize_pcd=False,
        )
    )

    # table data
    table_pcd_points = np.asarray(scene_pcd.points)[table_sim_mask]
    table_pcd_colors = np.asarray(scene_pcd.colors)[table_sim_mask]
    table_pcd_sim = table_attr["raw_similarity"][table_sim_mask].cpu().numpy()

    # # # # #
    # # # # # Plane Fitting for the Table
    # # # # #
    pcd_clus = o3d.geometry.PointCloud()
    pcd_clus.points = o3d.utility.Vector3dVector(table_pcd_points[:, :3])
    pcd_clus.colors = o3d.utility.Vector3dVector(table_pcd_colors)

    # plane
    plane_model, inliers = pcd_clus.segment_plane(
        distance_threshold=0.001, ransac_n=3, num_iterations=1000
    )
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # normal to the plane
    normal_plane = plane_model[:3]

    # threshold for the distance to the plane
    dist_plane_threshold = np.amax(
        np.abs(normal_plane @ table_pcd_points[inliers, :3].T + plane_model[-1])
    )

    if enable_visualization_pcd:
        inlier_cloud = pcd_clus.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd_clus.select_by_index(inliers, invert=True)
        fig = o3d.visualization.draw_plotly(
            [inlier_cloud, outlier_cloud],
        )
        fig.show()

    # target location
    target_data = dict()

    # similarity threshold
    threshold_targ = [0.95, 0.9]

    # filter radius
    filter_radius_targ = 0.05

    # clearance for the z-component at the target place point (m)
    target_z_clearance = 0.007

    for idx, obj in enumerate(targets):
        if idx != 1:
            continue

        # target
        target_centroid, target_z_bounds, _, target_sim_mask, target_attr = (
            get_centroid(
                nerf=nerf,
                env_pcd=env_pcd,
                pcd_attr=env_attr,
                positives=obj,
                threshold=threshold_targ[idx],
                filter_radius=filter_radius_targ,
                enable_convex_hull=True,
                enable_spherical_filter=True,
                visualize_pcd=False,
            )
        )

        # target data
        target_pcd_points = np.asarray(scene_pcd.points)[target_sim_mask]
        target_pcd_colors = np.asarray(scene_pcd.colors)[target_sim_mask]
        target_pcd_sim = target_attr["raw_similarity"][target_sim_mask].cpu().numpy()

        # # # # #
        # # # # # Desired Location for Placing the Objects
        # # # # #

        pcd_clus = o3d.geometry.PointCloud()
        pcd_clus.points = o3d.utility.Vector3dVector(target_pcd_points[:, :3])
        pcd_clus.colors = o3d.utility.Vector3dVector(target_pcd_colors)

        if enable_visualization_pcd:
            o3d.visualization.draw_plotly([pcd_clus])

        plane_model_target, inliers = pcd_clus.segment_plane(
            distance_threshold=0.001, ransac_n=3, num_iterations=1000
        )
        # [a, b, c, d] = plane_model_obj
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd_clus.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd_clus.select_by_index(inliers, invert=True)

        if enable_visualization_pcd:
            fig = o3d.visualization.draw_plotly(
                [inlier_cloud, outlier_cloud],
            )
            fig.show()

        # place-task desired location
        target_place_point = np.mean(target_pcd_points[inliers], axis=0)

        # incorporate the clearance for the target z-point
        target_place_point += target_z_clearance * normal_plane

        # store the data
        target_data[obj] = {"place_point": target_place_point}

    # # %%
    # # # # #
    # # # # # Scene Editing from the Gaussian Means: Generate a Sample Trajectory and Task
    # # # # #

    # objects: List[str] = ['saucepan', 'glass lid', 'knife', 'orange']
    # object_to_target: Dict[str, str] = {'saucepan': targets[0],
    #                                     'glass lid': targets[0],
    #                                     'knife': targets[1],
    #                                     'orange': targets[1]
    #                                     }

    # outputs for each object
    obj_outputs = {}

    # offset for each object
    offsets = np.zeros((len(objects), 3))

    # filter-size
    filter_radius = [0.055, 0.05, 0.05, 0.05]
    # TODO
    filter_radius.extend([filter_radius[-1]] * (len(objects) - len(filter_radius)))

    # similarity threshold
    threshold_obj = [0.95, 0.9, 0.9, 0.97]
    # TODO
    threshold_obj.extend([threshold_obj[-1]] * (len(objects) - len(threshold_obj)))

    for idx, obj in enumerate(objects):
        if idx < 2:
            continue

        print("*" * 50)
        print(f"Processing Object: {obj}")
        print("*" * 50)

        # prior information on the object masks
        obj_priors: Dict = {"mask_prior": table_attr["raw_similarity"]}

        # source location
        src_centroid, src_z_bounds, scene_pcd, similarity_mask, other_attr = (
            get_centroid(
                nerf=nerf,
                env_pcd=env_pcd,
                pcd_attr=env_attr,
                positives=objects[idx],
                negatives="object, things, stuff, texture",
                threshold=threshold_obj[idx],
                visualize_pcd=False,
                enable_convex_hull=True,
                enable_spherical_filter=True,
                enable_clustering=False,
                filter_radius=filter_radius[idx],
                obj_priors={},  # obj_priors,
                use_Mahalanobis_distance=True,
            )
        )

        # object
        object_pcd_points = np.asarray(scene_pcd.points)[similarity_mask]
        object_pcd_colors = np.asarray(scene_pcd.colors)[similarity_mask]
        object_pcd_sim = other_attr["raw_similarity"][similarity_mask].cpu().numpy()

        # if any(item in obj for item in ['pot', 'pan', 'lid']):
        # plane-fitting
        pcd_clus = o3d.geometry.PointCloud()
        pcd_clus.points = o3d.utility.Vector3dVector(object_pcd_points[:, :3])
        pcd_clus.colors = o3d.utility.Vector3dVector(object_pcd_colors)

        if enable_visualization_pcd:
            fig = o3d.visualization.draw_plotly([pcd_clus])
            fig.show()
        ## %%

        # TODO: Refactor into a function

        # distance threshold
        dist_threshold_plane_fitting = 0.008 if "knife" in obj else 0.001  # (default)

        plane_model_obj, inliers_obj = pcd_clus.segment_plane(
            distance_threshold=dist_threshold_plane_fitting,
            ransac_n=3,
            num_iterations=1000,
        )
        # [a, b, c, d] = plane_model_obj
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd_clus.select_by_index(inliers_obj)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd_clus.select_by_index(inliers_obj, invert=True)

        if enable_visualization_pcd or enable_minimal_visualization:
            print("*" * 50)
            print("Plane-Fitting: Stage 1")
            fig = o3d.visualization.draw_plotly(
                [inlier_cloud, outlier_cloud],
            )
            fig.show()
            print("*" * 50)

        if not "lid" in obj and not "knife" in obj:
            # # normal to the plane
            # normal_plane = plane_model[:3]

            # distance along the normal to the plane
            dist_plane = normal_plane @ object_pcd_points[:, :3].T + plane_model[-1]

            # threshold for the distance to the plane
            dist_plane_threshold_obj = dist_plane_threshold + (
                1e0 * dist_plane_threshold
            )
        elif "knife" in obj:
            # normal to the plane
            normal_plane_obj = plane_model_obj[:3]

            # distance along the normal to the plane
            dist_plane = (
                normal_plane_obj @ object_pcd_points[:, :3].T + plane_model_obj[-1]
            )

            # threshold for the distance to the plane
            dist_plane_threshold_obj = np.mean(
                np.abs(
                    normal_plane_obj @ object_pcd_points[inliers_obj, :3].T
                    + plane_model_obj[-1]
                )
            )
        elif "lid" in obj:
            # # # # #
            # # # # # Scene Editing for Elevated Objects: Two-Stage Process
            # # # # #

            # Stage 1

            # normal to the plane
            normal_plane_obj = plane_model_obj[:3]

            # distance along the normal to the plane
            dist_plane = (
                normal_plane_obj @ object_pcd_points[:, :3].T + plane_model_obj[-1]
            )

            # threshold for the distance to the plane
            dist_plane_threshold_obj = np.amax(
                np.abs(
                    normal_plane_obj @ object_pcd_points[inliers_obj, :3].T
                    + plane_model_obj[-1]
                )
            )
            dist_plane_threshold_obj = dist_plane_threshold_obj + (
                1e-1 * dist_plane_threshold_obj
            )

            # filter the points
            pcd_mask = dist_plane > dist_plane_threshold_obj
            vis_pcd_pts = object_pcd_points[:, :3][pcd_mask]
            vis_colors = object_pcd_colors[pcd_mask]

        # visualize the point cloud
        # mask for the selected points
        pcd_mask = dist_plane > dist_plane_threshold_obj
        if "knife" in obj:
            # include the inliers
            pcd_mask[inliers_obj] = True

        # selected points with colors
        vis_pcd_pts = object_pcd_points[:, :3][pcd_mask]
        vis_colors = object_pcd_colors[pcd_mask]

        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_pcd_pts)
        pcd.colors = o3d.utility.Vector3dVector(vis_colors)

        if enable_visualization_pcd:
            fig = o3d.visualization.draw_plotly([pcd])
            fig.show()

        # remove outliers
        pcd, inlier_ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)  # r=0.03

        if enable_visualization_pcd or enable_minimal_visualization:
            print("*" * 50)
            print("Post-Outlier-Removal")
            fig = o3d.visualization.draw_plotly([pcd])
            fig.show()
            print("*" * 50)

        if "knife" in obj:
            # compute the object's base
            obj_base = np.mean(pcd.points, axis=0)

            # modify the height of the base
            b_argmin = np.argmin(
                np.abs(
                    normal_plane_obj @ object_pcd_points[inliers_obj, :3].T
                    + plane_model_obj[-1]
                )
            )
            obj_base[-1] = object_pcd_points[inliers_obj, -1][b_argmin]

            # object's width
            knife_dim = np.max(pcd.points, axis=0) - np.min(pcd.points, axis=0)
        elif any(item in obj for item in fruits):
            # compute the object's base
            obj_base = np.mean(pcd.points, axis=0)

            # modify the height of the base
            base_pts = object_pcd_points[dist_plane < dist_plane_threshold_obj, :]
            obj_base[-1] = np.mean(base_pts[:, -1])

        # # Utilize Convex hull
        # if 'lid' in obj:
        #     # points classified as being part of the object
        #     pts_cond = np.asarray(pcd.points)

        #     # compute the convex hull
        #     convex_hull = ConvexHull(pts_cond)

        #     # examine the convex hull
        #     convex_hull_mask = in_convex_hull(np.asarray(scene_pcd.points), pts_cond[convex_hull.vertices])

        #     if print_debug_info:
        #         print(f'Convex Hull Proc. Before : {len(pts_cond)}, After: {len(convex_hull_mask.nonzero()[0])}')

        #     # update the similarity mask
        #     similarity_mask = np.logical_or(similarity_mask, convex_hull_mask)

        new_obj_mask = np.copy(similarity_mask)
        new_obj_mask[similarity_mask] = pcd_mask

        # # incorporate the mask after outlier removal
        out_rem_mask = np.zeros_like(new_obj_mask[new_obj_mask], dtype=bool)
        out_rem_mask[inlier_ind] = True
        new_obj_mask[new_obj_mask] = out_rem_mask

        # composite mask
        comp_mask = env_pcd_mask.clone()
        comp_mask[comp_mask == True] = torch.tensor(new_obj_mask).to(device)

        # print('*' * 50)
        # print(f'Num. points: {comp_mask.count_nonzero()}')
        # print('*' * 50)

        # # mask for the selected points based on opacity
        # opac_mask = comp_mask[env_pcd_mask]
        # opac_mask[(env_attr['opacities'] < 0.1).squeeze()] = False
        # comp_mask[env_pcd_mask] = opac_mask

        # print('*' * 50)
        # print(f'Num. points: {comp_mask.count_nonzero()}')
        # print('*' * 50)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(nerf.pipeline.model.means[comp_mask].clone().detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(nerf.pipeline.model.features_dc[comp_mask].clone().detach().cpu().numpy())

        # o3d.visualization.draw_plotly([pcd])

        # update the local point-cloud mask
        pcd_mask_update = np.zeros_like(pcd_mask[pcd_mask], dtype=bool)
        pcd_mask_update[inlier_ind] = True
        pcd_mask[pcd_mask] = pcd_mask_update

        # pcd_mask = torch.logical_and(pcd_mask, env_attr['opacities'][comp_mask]  > 0.1)

        # update the translation
        object_pcd_points_upd = np.asarray(
            nerf.pipeline.model.means[comp_mask].clone().detach().cpu().numpy()
        )
        object_pcd_colors_upd = np.asarray(
            nerf.pipeline.model.features_dc[comp_mask].clone().detach().cpu().numpy()
        )

        if any(item in obj for item in ["pot", "pan"]):
            # using the plane fitted to the rim of the pot
            dist_plane_inliers = (
                normal_plane @ object_pcd_points[inliers_obj, :3].T + plane_model[-1]
            )

            # height of the pot along the normal to the table
            pot_height = np.mean(dist_plane_inliers) - dist_plane_threshold_obj

        if any(item in obj for item in ["pot", "pan", "lid"]):
            # base of the object
            pcd_mask_base = (
                np.abs(dist_plane[pcd_mask] - dist_plane_threshold_obj)
                < 1e0 * dist_plane_threshold_obj
            )
            vis_pcd_pts = object_pcd_points_upd.copy()
            vis_colors = object_pcd_colors_upd.copy()

            vis_colors[pcd_mask_base, :3] = [1, 0, 0]

            # point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vis_pcd_pts)
            pcd.colors = o3d.utility.Vector3dVector(vis_colors)

            if enable_visualization_pcd:
                fig = o3d.visualization.draw_plotly([pcd])
                fig.show()

            # point cloud for the base of the pot
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                object_pcd_points_upd[pcd_mask_base, :3]
            )
            pcd.colors = o3d.utility.Vector3dVector(
                object_pcd_colors_upd[pcd_mask_base, :3]
            )

            # remove outliers
            pcd, inlier_ind = pcd.remove_radius_outlier(
                nb_points=3, radius=0.01
            )  # r=0.03

            if enable_visualization_pcd:
                o3d.visualization.draw_plotly([pcd])

            # the centroid the object's base
            obj_base = np.mean(pcd.points, axis=0)

        # target place-point
        target_place_point = target_data[object_to_target[obj]]["place_point"]

        if any(item in obj for item in ["pot", "pan"]):
            # translation
            translation = target_place_point - obj_base
        elif "lid" in obj:
            # translation
            translation = (target_place_point - obj_base) + pot_height * normal_plane
        elif "knife" in obj:
            # translation
            translation = target_place_point - obj_base
        elif any(item in obj for item in fruits):
            # translation
            translation = target_place_point - obj_base

            try:
                # translation (plus a further translation given the width of the knife)
                translation += 0.5 * np.array([knife_dim[0], 0, 0])
            except NameError:
                pass
        else:
            pass

        # outputs
        obj_outputs[obj] = {
            "centroid": src_centroid,
            "z_bounds": src_z_bounds,
            "pcd": scene_pcd,
            "similarity_mask": similarity_mask,
            "other_attr": other_attr,
            "translation": translation,
            "comp_mask": comp_mask,
        }

        # # %%
        # # # # #
        # # # # # Grasp Generation
        # # # # #

        from scipy.spatial import KDTree
        from utils.grasp_utils import *

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

        # output directory
        grasp_output_dir = Path(f"grasp_results/{obj}/{obj}_{trial_idx}")

        # point cloud and endpoints
        # sample points
        cloud_masked = np.asarray(object_pcd_points_upd)
        color_masked = np.asarray(object_pcd_colors_upd)

        # # # # #
        # # # # # Proposed Grasps from GraspNet
        # # # # #

        # proposed grasps from GraspNet
        cand_grasps, pcd = demo(cloud_masked, color_masked, cfgs)

        cand_grasps.nms()
        cand_grasps.sort_by_score()
        gg = cand_grasps

        # option to display the axes and gridlines
        showaxes_grid = False

        print("*" * 50)
        print("GraspNet without Affordance")
        print("*" * 50)

        # visualize the grasps
        fig_grasp_wout_aff = visualize_grasps(
            gg,
            pcd=pcd,
            num_vis_grasp=num_vis_grasp,
            grasp_group_color=grasp_group_color,
            showaxes_grid=showaxes_grid,
        )

        if enable_minimal_visualization:
            fig_grasp_wout_aff.show()

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

        print("*" * 50)
        print("GraspNet with Affordance")
        print("*" * 50)

        # visualize the grasps
        fig_grasp_wout_aff_wout_pp = visualize_grasps(
            ranked_grasps,
            pcd=pcd,
            num_vis_grasp=num_vis_grasp,
            grasp_group_color=grasp_group_color,
            showaxes_grid=showaxes_grid,
        )

        if enable_minimal_visualization:
            fig_grasp_wout_aff_wout_pp.show()

        print("*" * 50)
        print("GraspNet with Affordance and Heuristics")
        print("*" * 50)

        # # # # #
        # # # # # Reorient Proposed Grasps
        # # # # #

        # reoriented grasps
        reor_gp = reorient_grasps(ranked_grasps[:num_vis_grasp], normal_plane)

        fig_grasp_w_aff_w_pp = visualize_grasps(
            reor_gp,
            pcd=pcd,
            num_vis_grasp=num_vis_grasp,
            grasp_group_color=grasp_group_color,
            showaxes_grid=showaxes_grid,
        )

        if enable_minimal_visualization:
            fig_grasp_w_aff_w_pp.show()

        # # # # #
        # # # # # Save the Grasps.
        # # # # #

        # filename
        pose_filename = Path(f"{grasp_output_dir}/{obj}_graspnet.npy")

        # number of grasps to save
        num_grasps_save = num_vis_grasp

        # save the proposed grasps
        save_grasps(
            filename=pose_filename, grasps=cand_grasps, num_grasps_save=num_grasps_save
        )

        # filename
        pose_filename = Path(f"{grasp_output_dir}/{obj}_w_affordance.npy")

        # save the proposed grasps
        save_grasps(
            filename=pose_filename,
            grasps=ranked_grasps,
            num_grasps_save=num_grasps_save,
        )

        # filename
        pose_filename = Path(f"{grasp_output_dir}/{obj}_w_post_processing.npy")

        # save the proposed grasps
        save_grasps(
            filename=pose_filename, grasps=reor_gp, num_grasps_save=num_grasps_save
        )

        # save translation vector
        np.save(
            file=f"{grasp_output_dir}/translation.npy",
            arr=obj_outputs[objects[idx]]["translation"],
        )

        # # %%
        # # # # #
        # # # # # Scene Editing from the Gaussian Means: Generate a Sample Camera Trajectory
        # # # # #

        # output directory for videos
        video_output_dir = Path(f"render_results/{scene_name}/{trial_idx}")

        # desired rotation
        rotation_angle = [-5, 0, 5, 10]
        run_idx = 0
        des_rot_angle = rotation_angle[run_idx] * np.pi / 180
        des_rot_quat = (
            roma.rotvec_to_unitquat(des_rot_angle * torch.tensor(normal_plane))
            .to(device)
            .float()[None]
        )

        # origin of the rotation axis
        rot_origin = torch.tensor(obj_base, device=device).float()

        # TODO: Update the creation of the Cameras object.
        # cameras for rendering
        if any(item in obj for item in fruits | {"knife"}):
            # cameras for rendering
            cams_inds = list(reversed(range(4, 10)))
        else:
            # cameras for rendering
            cams_inds = list(reversed(range(4, 10)))

        cams_render = nerf.pipeline.datamanager.eval_dataset.cameras
        cams_render = Cameras(
            fx=cams_render.fx[cams_inds],
            fy=cams_render.fy[cams_inds],
            cx=cams_render.cx[cams_inds],
            cy=cams_render.cy[cams_inds],
            camera_type=cams_render.camera_type[cams_inds],
            camera_to_worlds=cams_render.camera_to_worlds[cams_inds],
        )

        # number of interpolation steps for the camera
        num_cam_interp_steps = 240 // (len(cams_render) - 1)

        # number of interpolation steps for the poses
        num_pose_interp_steps = num_cam_interp_steps * (len(cams_render) - 1)

        # composite mask
        comp_mask_obj = obj_outputs[objects[idx]]["comp_mask"]

        # translation
        translation = obj_outputs[objects[idx]]["translation"]

        # update the Gaussians
        # initial means of the Gaussians
        rel_means_a = nerf.pipeline.model.means.clone()

        # final means of the Gaussians
        rel_means_b = nerf.pipeline.model.means.clone()

        # update the means for the pertinent object
        rel_means_b[comp_mask_obj] = rel_means_a[comp_mask_obj] + torch.tensor(
            translation
        ).float().to(device)

        # update the quaternions
        rel_quats = nerf.pipeline.model.quats.clone()

        # interpolate the means and the quaternions
        means = torch.zeros((num_pose_interp_steps, comp_mask_obj.shape[0], 3)).to(
            device
        )
        quats = torch.zeros((num_pose_interp_steps, comp_mask_obj.shape[0], 4)).to(
            device
        )

        # update the Gaussians for the static scene
        means[:, torch.logical_not(comp_mask_obj), :] = torch.broadcast_to(
            rel_means_b[torch.logical_not(comp_mask_obj)],
            (
                num_pose_interp_steps,
                *rel_means_b[torch.logical_not(comp_mask_obj)].shape,
            ),
        )

        quats[:, torch.logical_not(comp_mask_obj), :] = torch.broadcast_to(
            rel_quats[torch.logical_not(comp_mask_obj)],
            (num_pose_interp_steps, *rel_quats[torch.logical_not(comp_mask_obj)].shape),
        )

        # desired final orientation
        des_quats = rel_quats

        # update the Gaussians for the pertinent object
        means[:, comp_mask_obj, :], quats[:, comp_mask_obj, :] = (
            get_interpolated_gaussians(
                rel_means_a[comp_mask_obj],
                rel_means_b[comp_mask_obj],
                rel_quats[comp_mask_obj],
                des_quats[comp_mask_obj],
                steps=num_pose_interp_steps,
                des_rot=des_rot_quat,
                rot_origin=rot_origin,
            )
        )

        means = means[:, comp_mask_obj, :]
        quats = quats[:, comp_mask_obj, :]

        # lazy imports
        import importlib
        import utils.render_utils

        importlib.reload(utils.render_utils)

        from utils.render_utils import RenderInterpolated

        # renderer
        scene_renderer = RenderInterpolated(
            load_config=config_path,
            output_path=Path(
                f'{video_output_dir}/scene_editing_eval_{"_".join(obj.split(" "))}.mp4'
            ),
            pose_source="eval",
            interpolation_steps=num_cam_interp_steps,
            output_format="video",
            order_poses=False,
            frame_rate=25,  # default
        )

        # render the scene
        scene_renderer.main(
            gaussian_means=means,
            gaussian_quats=quats,
            gaussian_mask=comp_mask_obj,
            cameras=cams_render,
            pipeline=nerf.pipeline,
        )

        # save the point cloud
        if save_multi_stage_pcd:
            # generate the point cloud of the environment
            env_pcd_ms, _, _ = nerf.generate_point_cloud(use_bounding_box=True)

            from utils.exporter_utils import post_process_point_cloud

            # post-process the point cloud
            std_ratio = 0.2
            pcd = post_process_point_cloud(pcd=env_pcd_ms, std_ratio=std_ratio)

            # create the director, if necessary
            pcd_output_dir_ms = Path(f"{pcd_output_dir}/multi_stage_{trial_idx}")
            pcd_output_dir_ms.mkdir(parents=True, exist_ok=True)

            # save the point cloud
            o3d.io.write_point_cloud(
                f"{pcd_output_dir_ms}/point_cloud_stage_{obj}.ply", pcd
            )

            if save_mesh:
                # create the directory, if necessary
                mesh_output_dir_ms = Path(f"{mesh_output_dir}/multi_stage_{trial_idx}")
                mesh_output_dir_ms.mkdir(parents=True, exist_ok=True)

                if use_alpha_mesh:
                    # alpha value
                    alpha_mesh = 0.01
                    mesh = (
                        o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                            pcd, alpha_mesh
                        )
                    )
                else:
                    # create a mesh from a sparse point cloud using the Poisson algorithm
                    CONSOLE.print("Computing Mesh... this may take a while.")
                    mesh, densities = (
                        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                            pcd, depth=9
                        )
                    )
                    vertices_to_remove = densities < np.quantile(densities, 0.1)
                    mesh.remove_vertices_by_mask(vertices_to_remove)
                    print("\033[A\033[A")
                    CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

                # save the mesh
                CONSOLE.print("Saving Mesh...")
                o3d.io.write_triangle_mesh(
                    f"{mesh_output_dir_ms}/poisson_mesh_stage_{obj}.ply", mesh
                )
                print("\033[A\033[A")
                CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

    # clear memory
    del nerf
    gc.collect()
    torch.cuda.empty_cache()
# %%
