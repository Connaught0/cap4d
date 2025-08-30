import numpy as np
from torch.utils.data import Dataset
import einops
import torchvision.utils as tvu
import torch
from cap4d.flame.flame import CAP4DFlameSkinner
from cap4d.datasets.utils import (
    load_frame, 
    crop_image, 
    rescale_image, 
    apply_bg, 
    load_flame_verts_and_cam,
    get_bbox_from_verts,
    load_camera_rays,
    verts_to_pytorch3d,
)


class CAP4DInferenceDataset(Dataset):
    def __init__(
        self, 
        resolution=512,
        downsample_ratio=8,
    ):
        self.resolution = resolution
        self.latent_resolution = self.resolution // downsample_ratio

        self.flame_skinner = CAP4DFlameSkinner(
            add_mouth=True, 
            n_shape_params=150,
            n_expr_params=65,
        )
        self.head_vertex_ids = np.genfromtxt("data/assets/flame/head_vertices.txt").astype(int)

        self.flame_list = None
        self.ref_extr = None
        self.data_path = None
        self.is_ref = False

    def __len__(self):
        assert self.flame_list is not None, "self.flame_list not properly initialized"
        return len(self.flame_list)

    def __getitem__(self, idx):
        flame_item = self.flame_list[idx]

        verts_2d, offsets_3d, intrinsics, extrinsics = load_flame_verts_and_cam(
            self.flame_skinner,
            flame_item,
            self.is_ref,
        )
        
        crop_box = get_bbox_from_verts(verts_2d, self.head_vertex_ids)
        flame_item["crop_box"] = crop_box

        if "img_dir_path" in flame_item:
            # we have images available, load them
            # load and crop image, including background
            img_dir_path = flame_item["img_dir_path"]
            timestep_id = flame_item["timestep_id"]
            img = load_frame(img_dir_path, timestep_id)
            vis_img = img.copy()
            tvu.save_image(torch.from_numpy(img).permute(2,0,1).float() / 255.0, f"debug_img_{timestep_id}.png")
            del flame_item["img_dir_path"]  # delete string from flame dict so that it can be collated
            if "bg_dir_path" in flame_item:
                bg = load_frame(flame_item["bg_dir_path"], timestep_id)
                del flame_item["bg_dir_path"]  # delete string from flame dict so that it can be collated
            else:
                print(f"WARNING: bg does not exist for image {img_dir_path}. Make sure the background is white.")
                bg = np.ones_like(img) * 255
            out_crop_mask = np.ones_like(img[..., [0]])
            img = apply_bg(img, bg)
            img = crop_image(img, crop_box, bg_value=255)
            # Create visualization overlay on image
 
            #                y1
            # crop box x1，        x2,
            #                y2

            import cv2
            # Draw all vertices as small dots
            # Note: OpenCV's origin (0,0) is at top-left, while OpenGL's is at bottom-left
            # So we need to flip the y-coordinate relative to image height
            h = vis_img.shape[0]
            for v in verts_2d:
                y = int(v[1])  # Flip y coordinate
                x = int(v[0])      # x coordinate doesn't need flipping
                cv2.circle(vis_img, (x, y), radius=1, color=(0,255,0), thickness=-1)
            
            # Draw head vertices as red dots
            for v in verts_2d[self.head_vertex_ids]:
                y = int(v[1])  # Flip y coordinate
                x = int(v[0])      # x coordinate doesn't need flipping
                cv2.circle(vis_img, (x, y), radius=2, color=(0,0,255), thickness=-1)
            
            # Draw crop box
            x1, y1, x2, y2 = crop_box
            # Flip y coordinates for crop box (x coordinates stay the same)
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color=(255,0,0), thickness=2)
            
            # Save visualization
            tvu.save_image(torch.from_numpy(vis_img).permute(2,0,1).float() / 255.0, f"debug_vis_{timestep_id}.png")
            tvu.save_image(torch.from_numpy(img).permute(2,0,1).float() / 255.0, f"debug_crop_img_{timestep_id}.png")
            out_crop_mask = crop_image(out_crop_mask, crop_box, bg_value=0)
            img = rescale_image(img, self.resolution)
            img = ((img / 127.5) - 1.0).astype(np.float32)
            out_crop_mask = rescale_image(out_crop_mask, self.latent_resolution)
            is_ref = True
        else:
            # no image available means these images need to be generated
            # set image to zero
            img = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)
            out_crop_mask = np.ones((self.latent_resolution, self.latent_resolution), dtype=np.float32)
            is_ref = False

        # load and transform ray map
        ray_map = load_camera_rays(
            crop_box,
            intrinsics,
            extrinsics,
            self.latent_resolution,
        )
        assert self.ref_extr is not None, "reference extrinsics ref_extr not set"
        # transform raymap to base extrinsics
        ray_map_h = ray_map.shape[1]
        ray_map = einops.rearrange(ray_map, 'v h w -> v (h w)')
        ray_map = self.ref_extr[:3, :3] @ ray_map
        ray_map = einops.rearrange(ray_map, 'v (h w) -> v h w', h=ray_map_h)

        # reference mask is one for reference dataset
        reference_mask = np.ones_like(out_crop_mask) * is_ref

        # convert pixel space vertices to pytorch3d space [-1, 1]
        verts_2d = verts_to_pytorch3d(verts_2d, np.array(crop_box))

        cond_dict = {
            "out_crop_mask": out_crop_mask[None],
            "reference_mask": reference_mask[None],
            "ray_map": ray_map[None],
            "verts_2d": verts_2d[None],
            "offsets_3d": offsets_3d[None],
        }  # [None] is for fake time dimension

        out_dict = {
            "jpg": img[None],  # jpg names comes from controlnet implementation
            "hint": cond_dict,
            "flame_params": flame_item,
        }
        
        return out_dict
