import torch
import argparse
from nerf.utils import *
from PIL import Image
from nerf_pose_estimate.dataset import NerfDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

def load_checkpoint(model, device, checkpoint_path):
    checkpoint_list = sorted(glob.glob(f'{checkpoint_path}/ngp_ep*.pth'))
    if checkpoint_list:
        checkpoint = checkpoint_list[-1]
    else:
        raise RuntimeError

    checkpoint_dict = torch.load(checkpoint, map_location=device)

    model.load_state_dict(checkpoint_dict['model'], strict=False)

    if model.cuda_ray:
        if 'mean_count' in checkpoint_dict:
            model.mean_count = checkpoint_dict['mean_count']
        if 'mean_density' in checkpoint_dict:
            model.mean_density = checkpoint_dict['mean_density']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--epoch', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()
    opt.test = True

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(opt)

    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

    model.to(device)
    load_checkpoint(model, device, checkpoint_path=f"{opt.workspace}/checkpoints")
    model.eval()

    dataset = NerfDataset(device=device, path=opt.path, scale=opt.scale, offset=opt.offset)
    intrinsics = dataset.intrinsics
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    save_path = os.path.join(opt.workspace, 'results')
    os.makedirs(save_path, exist_ok=True)

    to_pil_image = ToPILImage()

    with torch.no_grad():

        for i, data in enumerate(dataloader):

            with torch.cuda.amp.autocast(enabled=opt.fp16):
                H, W = int(data['H']), int(data['W'])
                gt_pose = data["gt_pose"].to(device)
                gt_image = data["gt_image"].to(device)

                bg_color = None
                if bg_color is not None:
                    bg_color = bg_color.to(device)

                rays = get_rays(gt_pose, intrinsics, H, W)
                rays_o = rays['rays_o'] # [B, N, 3]
                rays_d = rays['rays_d'] # [B, N, 3]

                outputs = model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(opt))

                preds = outputs['image'].reshape(-1, H, W, 3)
                preds_depth = outputs['depth'].reshape(-1, H, W)

            pred = preds[0].detach().cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            pred_img = Image.fromarray(pred)

            gt_img = to_pil_image(gt_image.squeeze(0))

            plt.subplot(1,2,1)
            plt.imshow(pred_img)
            plt.gca().set_title("prediction")
            plt.subplot(1,2,2)
            plt.imshow(gt_img)
            plt.gca().set_title("ground truth")
            plt.show()