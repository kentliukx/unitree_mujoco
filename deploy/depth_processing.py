"""Shared RealSense depth preprocessing for deployment and debug tools."""

import numpy as np


def resize_depth(depth, cfg):
    crop_cols = max(int(getattr(cfg, "realsense_crop_cols_each_side", 0)), 0)
    src_h, src_w = depth.shape
    dst_h = int(cfg.depth_height)
    dst_w = int(cfg.depth_width)
    resize_w = dst_w + 2 * crop_cols
    if src_h == dst_h and src_w == dst_w:
        return depth

    if src_h % dst_h == 0 and src_w % resize_w == 0:
        block_h = src_h // dst_h
        block_w = src_w // resize_w
        blocks = depth.reshape(dst_h, block_h, resize_w, block_w)
        sample_n = max(int(getattr(cfg, "realsense_center_sample_size", 1)), 1)
        sample_h = min(sample_n, block_h)
        sample_w = min(sample_n, block_w)
        y0 = max((block_h - sample_h) // 2, 0)
        x0 = max((block_w - sample_w) // 2, 0)
        center = blocks[:, y0:y0 + sample_h, :, x0:x0 + sample_w]
        valid = center > 0.0
        mode = str(getattr(cfg, "realsense_downsample_mode", "mean")).lower()
        if mode == "mean":
            summed = np.where(valid, center, 0.0).sum(axis=(1, 3))
            counts = valid.sum(axis=(1, 3))
            resized = np.where(counts > 0, summed / np.maximum(counts, 1), cfg.depth_max)
        elif mode == "min":
            valid_depth = np.where(valid, center, cfg.depth_max)
            resized = valid_depth.min(axis=(1, 3))
        else:
            raise ValueError(f"Unsupported realsense_downsample_mode: {mode}")
        return crop_depth_columns(resized, crop_cols)

    y_idx = np.linspace(0, src_h - 1, dst_h).astype(np.int32)
    x_idx = np.linspace(0, src_w - 1, resize_w).astype(np.int32)
    return crop_depth_columns(depth[np.ix_(y_idx, x_idx)], crop_cols)


def preprocess_depth(depth, cfg):
    depth = resize_depth(depth, cfg)
    depth = np.nan_to_num(
        depth,
        nan=cfg.depth_max,
        posinf=cfg.depth_max,
        neginf=cfg.depth_max,
    )
    depth[depth <= 0.0] = cfg.depth_max
    return np.clip(depth, cfg.depth_min, cfg.depth_max).astype(np.float32, copy=False)


def crop_depth_columns(depth, crop_cols):
    if crop_cols <= 0:
        return depth
    return depth[:, crop_cols:-crop_cols]
