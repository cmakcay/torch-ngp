from nerf.utils import get_rays
from nerf.provider import nerf_matrix_to_ngp, rand_poses
from torch.utils.data import Dataset
import numpy as np

class NerfDataset(Dataset):
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
