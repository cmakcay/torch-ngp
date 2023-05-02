from nerf.utils import get_rays
from nerf.provider import nerf_matrix_to_ngp, rand_poses
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
from torchvision.io import read_image
import torch

def load_data(path, scale, offset):
    if type(path) == str:
        path = Path(path)

    assert (path / "transforms.json").exists(), "transforms.json not found in data dir"

    with open(path / "transforms.json", "r") as f:
        metadata = json.load(f)

    # load image size
    if 'h' in metadata and 'w' in metadata:
        H = int(metadata['h'])
        W = int(metadata['w'])
    else:
        # we have to actually read an image to get H and W later.
        H = W = None

    # Read Images
    frame_list = metadata["frames"]
    frames = list()
    poses = list()
    img_extension = None
    has_alpha_channel = None

    for cur_frame in frame_list:
        cur_file_path = path / cur_frame["file_path"]
        cur_pose = cur_frame["transform_matrix"]

        # Decide on the extension
        if img_extension is None:
            if (path / f"{cur_file_path}.png").exists():
                img_extension = ".png"
            elif (path / f"{cur_file_path}.jpg").exists():
                img_extension = ".jpg"
            else:
                raise NotImplementedError
        cur_file_path = path / f"{cur_file_path}{img_extension}"

        # Read the image
        cur_img = read_image(str(cur_file_path))
        _c, _h, _w = cur_img.shape
        if H is None or W is None:
            H = _h
            W = _w
        assert H == _h and W == _w
        if has_alpha_channel is None:
            if _c == 3:
                has_alpha_channel = False
            elif _c == 4:
                has_alpha_channel = True
            else:
                raise NotImplementedError
        if has_alpha_channel:
            assert _c == 4
        else:
            assert _c == 3
        frames.append(cur_img)

        # Read the pose
        cur_pose = np.array(cur_pose, dtype=np.float32) # [4, 4]
        cur_pose = nerf_matrix_to_ngp(cur_pose, scale=scale, offset=offset)
        poses.append(cur_pose)

    poses = torch.from_numpy(np.stack(poses, axis=0)) # [N, 4, 4]

    print(f"\n Info: the dataset has been loaded with {len(poses)} samples. \n")

    # load intrinsics
    if 'fl_x' in metadata or 'fl_y' in metadata:
        fl_x = (metadata['fl_x'] if 'fl_x' in metadata else metadata['fl_y'])
        fl_y = (metadata['fl_y'] if 'fl_y' in metadata else metadata['fl_x'])
    elif 'camera_angle_x' in metadata or 'camera_angle_y' in metadata:
        fl_x = W / (2 * np.tan(metadata['camera_angle_x'] / 2)) if 'camera_angle_x' in metadata else None
        fl_y = H / (2 * np.tan(metadata['camera_angle_y'] / 2)) if 'camera_angle_y' in metadata else None
        if fl_x is None: fl_x = fl_y
        if fl_y is None: fl_y = fl_x
    else:
        raise RuntimeError('Failed to load focal length, please check the transforms.json!')

    cx = metadata['cx'] if 'cx' in metadata else W / 2
    cy = metadata['cy'] if 'cy' in metadata else H / 2

    intrinsics = np.array([fl_x, fl_y, cx, cy])

    return(frames, poses, intrinsics, H, W)


class NerfDataset(Dataset):
    def __init__(self, device, path, scale, offset, num_rays=None, random_poses=False, random_poses_args=None) -> None:
        super().__init__()

        self.device = device
        self.num_rays = num_rays
        self.frames, self.poses, self.intrinsics, self.H, self.W = load_data(path, scale, offset)

    def __len__(self):
        assert len(self.poses) == len(self.frames)
        return len(self.poses)

    def __getitem__(self, index):
        output = {
            "gt_image": self.frames[index],
            "gt_pose": self.poses[index],
            "H": self.H,
            "W": self.W
        }
        return output

# TODO
class NerfRandomDataset(Dataset):
    def __init__(self, device, intrinsics, image_size, num_rays, random_poses=False, random_poses_args=None) -> None:
        super().__init__()

        self.device = device

        self.intrinsics = intrinsics # [fl_x, fl_y, cx, cy]
        assert len(image_size) == 2
        self.H, self.W = image_size
        self.num_rays = num_rays

        # Generate random poses on a unit sphere
        self.random_poses = random_poses
        if self.random_poses:
            self.poses = rand_poses(
                size = random_poses_args["num_poses"],
                device = self.device,
                radius = random_poses_args["radius"],
                theta_range = random_poses_args["theta_range"],
                phi_range = random_poses_args["theta_range"],
            )

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        cur_pose = self.poses[index]
        s = np.sqrt(self.H * self.W / self.num_rays)
        rH, rW = int(self.H / s), int(self.W / s)
        rays = get_rays(cur_pose, self.intrinsics / s, rH, rW, -1)

        return rays
