import json
import os
from pathlib import Path
import gc
import numpy as np
import torch
from sagesplat.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import torchvision
import copy
import argparse
import os
import random
import numpy as np
import torch
import io
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib
import pdb

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from lang_sam import LangSAM
from sagesplat.vrb.networks.model import VRBModel
from sagesplat.vrb.networks.traj import TrajAffCVAE


class AffordanceDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomGrayscale(p=0.05),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.3, saturation=0.3, hue=0.3
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        super().__init__(cfg, device, image_list, cache_path)

    def __call__(self, image_idx):
        # image_idx: index of the image in the training dataset
        output = self.data[image_idx].type(torch.float32).to(self.device)
        return output

    def compute_heatmap(self, points, image_size, k_ratio=3.0):
        points = np.asarray(points)
        heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
        n_points = points.shape[0]
        for i in range(n_points):
            x = points[i, 0]
            y = points[i, 1]
            col = int(x)
            row = int(y)
            try:
                heatmap[col, row] += 1.0
            except:
                col = min(max(col, 0), image_size[0] - 1)
                row = min(max(row, 0), image_size[1] - 1)
                heatmap[col, row] += 1.0
        k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
        if k_size % 2 == 0:
            k_size += 1
        heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        heatmap = heatmap.transpose()
        return heatmap

    def run_inference(self, image_pil):

        # # chopping scene
        objects = ["knife", "fruit", "orange", "chopping board", "cutting board"]

        bboxes = []
        for obj in objects:
            with torch.no_grad():
                masks, boxes, phrases, logits = self.langSAM.predict(image_pil, obj)
            bboxes.append(boxes)

        contact_points = []
        pred_contact_points = []
        trajectories = []
        mixtures = []
        for boxes in bboxes:
            if boxes.shape[0] > 0:
                # box = boxes[0]
                for box in boxes:
                    y1, x1, y2, x2 = box

                    # bbox_offset = 20
                    bbox_offset = 0
                    y1, x1, y2, x2 = (
                        int(y1) - bbox_offset,
                        int(x1) - bbox_offset,
                        int(y2) + bbox_offset,
                        int(x2) + bbox_offset,
                    )

                    width = y2 - y1
                    height = x2 - x1

                    diff = width - height
                    if width > height:
                        y1 += int(diff / np.random.uniform(1.5, 2.5))
                        y2 -= int((diff / (np.random.uniform(1.5, 2.5) + diff % 2)))
                    else:
                        diff = height - width
                        x1 += int(diff / np.random.uniform(1.5, 2.5))
                        x2 -= int((diff / (np.random.uniform(1.5, 2.5) + diff % 2)))

                    img = np.asarray(image_pil)
                    input_img = img[x1:x2, y1:y2]
                    if input_img.shape[0] == 0 or input_img.shape[1] == 0:
                        continue

                    inp_img = Image.fromarray(input_img)
                    # inp_img = transform(inp_img).unsqueeze(0).half().to(torch.device('cuda:0'))
                    inp_img = (
                        self.transform(inp_img).unsqueeze(0).to(torch.device("cuda:0"))
                    )
                    gm = GaussianMixture(n_components=3, covariance_type="diag")
                    centers = []
                    trajs = []
                    traj_scale = 0.1
                    with torch.no_grad():
                        ic, pc = self.affVRBnet.inference(inp_img, None, None)
                        pc = pc.cpu().numpy()
                        ic = ic.cpu().numpy()
                        i = 0
                        w, h = input_img.shape[:2]
                        sm = pc[i, 0] * np.array([h, w])
                        centers.append(sm)
                        trajs.append(ic[0, 2:])
                    gm.fit(np.vstack(centers))
                    mixtures.append(sm)
                    # cp, indx = gm.sample(50)
                    cp, indx = gm.sample(50)
                    x2, y2 = np.vstack(trajs)[np.random.choice(len(trajs))]
                    dx, dy = (
                        np.array([x2, y2]) * np.array([h, w])
                        + np.random.randn(2) * traj_scale
                    )
                    scale = 40 / max(abs(dx), abs(dy))
                    adjusted_cp = np.array([y1, x1]) + cp
                    contact_points.append(adjusted_cp)
                    trajectories.append([x2, y2, dx, dy])
                    # mixtures.append(copy.deepcopy(gm))

                    pred_contact_points.append(np.array([y1, x1]) + np.vstack(centers))

        original_img = np.asarray(image_pil)

        # image dimensions
        H, W, _ = original_img.shape

        # option to pose the problem as a binary classification problem
        use_binary_classification = False
        use_density = False
        get_contact_directions = True

        if len(contact_points) == 0:
            # heatmap
            im = 1e-8 * np.ones((H, W, 1))
            contact_directions = np.zeros((*original_img.shape[:2], 3))

            return im, contact_directions

        ############### MIXTURE MODEL DENSITY LEARNING ##########################
        contact_points_rs = np.concatenate(contact_points, axis=0)
        # contact_points_rs = np.concatenate(pred_contact_points, axis=0)
        gm_cp = GaussianMixture(
            n_components=len(contact_points_rs), covariance_type="diag"
        )
        gm_cp.fit(np.vstack(contact_points_rs))

        if use_binary_classification:

            # predicted contact points indices
            pred_contact_points = np.concatenate(pred_contact_points, axis=0).reshape(
                (-1, 2)
            )
            pred_contact_points_ceil = np.ceil(pred_contact_points)

            # enforce bounds
            pred_contact_points_ceil[pred_contact_points_ceil[:, 0] > W - 1, 0] = W - 1
            pred_contact_points_ceil[pred_contact_points_ceil[:, 1] > H - 1, 1] = H - 1

            pred_contact_points_floor = np.floor(pred_contact_points)
            pred_contact_points_ceil_x_floor_y = np.stack(
                [pred_contact_points_ceil[:, 0], pred_contact_points_floor[:, 1]],
                axis=1,
            )
            pred_contact_points_floor_x_ceil_y = np.stack(
                [pred_contact_points_floor[:, 0], pred_contact_points_ceil[:, 1]],
                axis=1,
            )

            pred_contact_points_ind = np.vstack(
                (
                    pred_contact_points_ceil,
                    pred_contact_points_floor,
                    pred_contact_points_ceil_x_floor_y,
                    pred_contact_points_floor_x_ceil_y,
                )
            ).astype(int)

            # generate labels
            contact_labels = np.zeros(original_img.shape[:2]).astype(np.int8)
            contact_labels[
                pred_contact_points_ind[:, 1], pred_contact_points_ind[:, 0]
            ] = 1

            return contact_labels[:, :, None]
        else:
            # Now that we have a distribution for this view we need to get the density at each pixel :grimacing:
            if len(contact_points) > 0:
                if use_density:
                    X, Y = np.meshgrid(np.arange(0, W), np.arange(0, H))
                    pts = np.stack([X, Y], axis=-1)  # H x W x 2
                    pts = pts.reshape(-1, 2)  # HW x 2
                    probs = gm_cp.predict_proba(pts).dot(gm_cp.weights_)  # Check shape
                    im = probs.reshape(H, W)
                else:
                    # heatmap
                    hmap = self.compute_heatmap(
                        np.vstack(contact_points),
                        (original_img.shape[1], original_img.shape[0]),
                        k_ratio=6,
                    )

                    im = hmap[:, :, None]

                    if get_contact_directions:
                        # get contact directions (x, y, is_contact)
                        contact_directions = np.zeros((*original_img.shape[:2], 3))

                        # default contact direction
                        default_contact_dir = np.array([1, 0])[None, None, :]

                        # initialize
                        contact_directions[:, :, :2] = default_contact_dir

                        for i, cp in enumerate(contact_points):
                            x2, y2, dx, dy = trajectories[i]
                            scale = 60 / max(abs(dx), abs(dy))

                            x, y = cp[:, 0], cp[:, 1]

                            x = int(np.mean(x))
                            y = int(np.mean(y))

                            # set contact direction
                            contact_directions[y, x, :2] = np.squeeze(
                                normalize(np.array([dx, dy])[:, None], axis=0)
                            )

                            # activate contact flag
                            contact_directions[y, x, -1] = 1
                    else:
                        contact_directions = None
            else:
                im = 1e-8 * np.ones((H, W, 1))

            return im, contact_directions

    def create(self, image_list):
        self.langSAM = LangSAM()
        self.affhand_head = TrajAffCVAE(
            in_dim=2 * 5,
            hidden_dim=192,
            latent_dim=4,
            condition_dim=256,
            coord_dim=64,
            traj_len=5,
        )

        self.affVRBnet = VRBModel(
            src_in_features=512,
            num_patches=1,
            hidden_dim=192,
            hand_head=self.affhand_head,
            encoder_time_embed_type="sin",
            num_frames_input=10,
            resnet_type="resnet18",
            embed_dim=256,
            coord_dim=64,
            num_heads=8,
            enc_depth=6,
            attn_kp=1,
            attn_kp_fc=1,
            n_maps=5,
        )

        # path to AFFLERF
        parent_path = Path(__file__).parent.parent.parent.resolve()

        dt = torch.load(
            f"{parent_path}/vrb/models/model_checkpoint_1249.pth.tar",
            map_location=torch.device("cuda:0"),
        )

        self.affVRBnet.load_state_dict(dt)
        self.affVRBnet.to(torch.device("cuda:0"))

        self.data = []
        rgb_image_transform = torchvision.transforms.ToPILImage()
        for i in tqdm(range(image_list.size()[0])):
            rgb_image = image_list[i, :, :, :]
            rgb_image = rgb_image_transform(rgb_image).convert("RGB")
            masked_hmap, contact_dir = self.run_inference(rgb_image)

            hmap_con_dir = np.concatenate((masked_hmap, contact_dir), axis=-1)
            hmap_con_dir = torch.from_numpy(hmap_con_dir)

            self.data.append(hmap_con_dir)

        self.data = torch.stack(self.data, dim=0)
        del self.langSAM
        del self.affVRBnet
        torch.cuda.empty_cache()
        gc.collect()


# %%
