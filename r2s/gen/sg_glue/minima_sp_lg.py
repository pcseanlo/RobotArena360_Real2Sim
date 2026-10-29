"""SuperPoint + LightGlue (MINIMA weights) matching used by ``inputs.place``.

Vendored from scene_generation:
  - ``src/utils/mini_utils.py``: ``run_pose_benchmark`` (``"sp_lg"`` branch only),
    ``test_relative_pose_demo``, ``eval_relapose`` (``save_figs`` is always False
    there, so the figure helpers are dropped);
  - ``src/utils/MINIMA/load_model.py``: ``load_sp_lg`` (``test_orginal_megadepth=False``);
  - ``src/utils/MINIMA/data_io_sp_lg.py``: ``DataIOWrapper``;
  - ``src/config/default.py``: the ``TEST`` block (img resize 640, df 8, no padding).

The third-party LightGlue code stays external: it is imported as
``third_party.LightGlue.lightglue`` from ``paths.scene_generation`` (the worker
appends that folder to ``sys.path``), exactly the module production used.
SuperPoint/LightGlue base weights come from the torch hub cache; the MINIMA
LightGlue checkpoint from ``<scene_generation>/MINIMA/weights/minima_lightglue.pth``.

Deviation: the old code rebuilt the matcher (and reloaded both checkpoints) for
every one of the 240 rendered views. Here it is built once per checkpoint and
reused; the weights are identical, so matches are unchanged.
"""
import os
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# src/config/default.py: _CN.TEST, lower-cased as lower_config() would return it.
TEST_CONFIG = {
    "img0_resize": 640,
    "img1_resize": 640,
    "df": 8,
    "padding": False,
    "coarse_scale": 0.125,
}


class DataIOWrapper(nn.Module):
    """
    Pre-propcess data from different sources
    """

    def __init__(self, model, config, ckpt=None):
        super().__init__()

        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)
        self.model = model
        self.config = config
        self.img0_size = config['img0_resize']
        self.img1_size = config['img1_resize']
        self.df = config['df']
        self.padding = config['padding']
        self.coarse_scale = config['coarse_scale']

        self.model = self.model.eval().to(self.device)

    def preprocess_image(self, img, device, resize=None, df=None, padding=None, cam_K=None, dist=None, gray_scale=True):
        # xoftr takes grayscale input images
        if gray_scale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape[:2]
        new_K = None
        img_undistorted = None
        if cam_K is not None and dist is not None:
            new_K, roi = cv2.getOptimalNewCameraMatrix(cam_K, dist, (w, h), 0, (w, h))
            img = cv2.undistort(img, cam_K, dist, None, new_K)
            img_undistorted = img.copy()

        if resize is not None:
            scale = resize / max(h, w)
            w_new, h_new = int(round(w * scale)), int(round(h * scale))
        else:
            w_new, h_new = w, h

        if df is not None:
            w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])

        img = cv2.resize(img, (w_new, h_new))
        scale = np.array([w / w_new, h / h_new], dtype=float)

        if padding:  # padding
            pad_to = max(h_new, w_new)
            img, mask = self.pad_bottom_right(img, pad_to, ret_mask=True)
            mask = torch.from_numpy(mask).to(device)
        else:
            mask = None
        # img = transforms.functional.to_tensor(img).unsqueeze(0).to(device)
        if len(img.shape) == 2:  # grayscale image
            img = torch.from_numpy(img)[None][None].to(self.device).float() / 255.0
        else:  # Color image
            img = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
        return img, scale, mask, new_K, img_undistorted

    def from_cv_imgs(self, img0, img1, K0=None, K1=None, dist0=None, dist1=None):
        # print('self.padding', self.padding)
        img0_tensor, scale0, mask0, new_K0, img0_undistorted = self.preprocess_image(
            img0, self.device, resize=self.img0_size, df=self.df, padding=self.padding, cam_K=K0, dist=dist0)
        img1_tensor, scale1, mask1, new_K1, img1_undistorted = self.preprocess_image(
            img1, self.device, resize=self.img1_size, df=self.df, padding=self.padding, cam_K=K1, dist=dist1)
        mkpts0, mkpts1, mconf, match_time = self.match_images(img0_tensor, img1_tensor, mask0, mask1)

        mkpts0 = mkpts0 * scale0
        mkpts1 = mkpts1 * scale1

        matches = np.concatenate([mkpts0, mkpts1], axis=1)
        data = {'matches': matches,
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
                'mconf': mconf,
                'img0': img0,
                'img1': img1,
                'match_time': match_time
                }
        if K0 is not None and dist0 is not None:
            data.update({'new_K0': new_K0, 'img0_undistorted': img0_undistorted})
        if K1 is not None and dist1 is not None:
            data.update({'new_K1': new_K1, 'img1_undistorted': img1_undistorted})
        return data

    def from_paths(self, img0_pth, img1_pth, K0=None, K1=None, dist0=None, dist1=None, read_color=False):

        imread_flag = cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE

        img0 = cv2.imread(img0_pth, imread_flag)
        img1 = cv2.imread(img1_pth, imread_flag)
        return self.from_cv_imgs(img0, img1, K0=K0, K1=K1, dist0=dist0, dist1=dist1)

    def match_images(self, image0, image1, mask0, mask1):
        batch = {'image0': image0, 'image1': image1}
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            batch.update({'mask0': ts_mask_0.unsqueeze(0), 'mask1': ts_mask_1.unsqueeze(0)})

        torch.cuda.synchronize()
        start = time.time()
        pred = self.model(batch)
        torch.cuda.synchronize()
        match_1 = time.time()
        match_time = match_1 - start

        mkpts0 = pred['keypoints0'].cpu().numpy()
        mkpts1 = pred['keypoints1'].cpu().numpy()
        matching_scores0 = pred['matching_scores'].detach().cpu().numpy()

        mconf = matching_scores0

        return mkpts0, mkpts1, mconf, match_time

    def pad_bottom_right(self, inp, pad_size, ret_mask=False):
        assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
        mask = None
        if inp.ndim == 2:
            padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
            padded[:inp.shape[0], :inp.shape[1]] = inp
            if ret_mask:
                mask = np.zeros((pad_size, pad_size), dtype=bool)
                mask[:inp.shape[0], :inp.shape[1]] = True
        elif inp.ndim == 3:
            padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
            padded[:, :inp.shape[1], :inp.shape[2]] = inp
            if ret_mask:
                mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
                mask[:, :inp.shape[1], :inp.shape[2]] = True
        else:
            raise NotImplementedError()
        return padded, mask


def load_sp_lg(args, test_orginal_megadepth=False):
    from third_party.LightGlue.lightglue import LightGlue, SuperPoint
    from third_party.LightGlue.lightglue.utils import rbd

    class Matching(torch.nn.Module):
        def __init__(self, sp_conf, lg_conf):
            super().__init__()
            device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
            self.extractor = SuperPoint(**sp_conf).eval().to(device)  # load the feature extractor
            self.matcher = LightGlue(features='superpoint', **lg_conf).eval().to(device)  # load the matcher
            n_layers = lg_conf['n_layers']
            # print(f"n_layers: {n_layers}")
            ckpt_path = args.ckpt
            # rename old state dict entries
            state_dict = torch.load(ckpt_path, map_location=device)
            for i in range(n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.matcher.load_state_dict(state_dict, strict=False)

        def forward(self, batch):
            # batch = {'image0': image0, 'image1': image1}
            image0 = batch['image0']
            image1 = batch['image1']
            # extract local features
            if test_orginal_megadepth:
                feats0 = self.extractor.extract(image0, resize=None)  # auto-resize the image, disable with resize=None
                feats1 = self.extractor.extract(image1, resize=None)
            else:
                feats0 = self.extractor.extract(image0)  # auto-resize the image, disable with resize=None
                feats1 = self.extractor.extract(image1)

            # match the features
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
            matches = matches01['matches']  # indices with shape (K,2)
            points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
            points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
            matching_scores0 = matches01['matching_scores0']
            matching_scores = matching_scores0[matches[..., 0]]

            return {'matching_scores': matching_scores, 'keypoints0': points0, 'keypoints1': points1}

    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }
    lg_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }
    matcher = Matching(sp_conf, lg_conf)
    matcher = DataIOWrapper(matcher, config=TEST_CONFIG)
    return matcher


_MATCHERS = {}


def load_model(method, args, use_path=True, test_orginal_megadepth=False):
    if method != "sp_lg":
        raise ValueError(f"only the sp_lg matcher is vendored, got {method!r}")
    key = (method, args.ckpt, use_path, test_orginal_megadepth)
    if key not in _MATCHERS:
        matcher = load_sp_lg(args, test_orginal_megadepth=test_orginal_megadepth)
        _MATCHERS[key] = matcher.from_paths if use_path else matcher.from_cv_imgs
    return _MATCHERS[key]


def eval_relapose(
        matcher,
        pair,
        save_figs,
        figures_dir=None,
        method=None,
):
    # Eval on pair
    im0 = pair['im0']
    im1 = pair['im1']

    match_res = matcher(im0, im1)
    img0_color = cv2.imread(im0)
    img1_color = cv2.imread(im1)
    img0_color = cv2.cvtColor(img0_color, cv2.COLOR_BGR2RGB)
    img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)


    mkpts0 = match_res['mkpts0']
    mkpts1 = match_res['mkpts1']
    mconf = match_res['mconf']
    if len(mkpts0) <= 10:
        return None, None, 0

    if len(mconf) > 0:
        conf_min = mconf.min()
        conf_max = mconf.max()
        mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)

    if len(mkpts0) >= 4:
        ret_H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
    else:
        inliers = None
        ret_H = None
    print(f"Number of inliers: {inliers.sum() if inliers is not None else 0}")
    return mkpts0, mkpts1, mkpts0.shape[0]


def test_relative_pose_demo(
        method="xoftr",
        save_dir=None,
        save_figs=False,
        args=None

):
    # Load pairs
    scene_pairs = {'im0': args.fig1, 'im1': args.fig2}

    # Load method
    matcher = load_model(method, args)
    # Eval
    keypoints0, keypoints1, points = eval_relapose(
        matcher,
        scene_pairs,
        save_figs=save_figs,
        figures_dir=save_dir,
        method=method,
    )
    return keypoints0,keypoints1,points


def run_pose_benchmark(method, fig1 , fig2, save_dir, checkpoint_dir):
    import argparse

    args = argparse.Namespace(
        method=method,
        exp_name="VisSYN",
        fig1=fig1,
        fig2=fig2,
        save_dir=save_dir,
        save_figs=False,
        checkpoint_dir = checkpoint_dir
    )
    checkpoint_dir = args.checkpoint_dir
    if method == "sp_lg":
        args.ckpt = os.path.join(checkpoint_dir, "minima_lightglue.pth")
    else:
        raise ValueError(f"Unknown method: {method}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        keypoints0,keypoints1,points = test_relative_pose_demo(
            method,
            save_dir=args.save_dir,
            save_figs=args.save_figs,
            args = args
        )
    return keypoints0, keypoints1, points
