Below is a drop-in style implementation that matches your pattern (state collected in `validate_img`, then actual work in `forward_raw`). It does:

1. **Sliding tiles (640)** with **overlap** (default 0.2)
2. **Pad** each tile to **640×640** (right/bottom) if it’s smaller
3. Stores per-tile metadata (offsets + valid area before padding) into `meta` so you can later do **decode_boxes → clip-to-valid-area → add offset → global NMS**

```python
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# assumes your framework provides these
# from your_pkg import ImageMatProcessor, ImageMatInfo


class SlidingTilePad640(ImageMatProcessor):
    """
    - Input:  HxW or HxWxC numpy image(s)
    - Output: flattened list of 640x640 tiles (padded if needed)
    - Meta:   tile offsets and each tile's valid (pre-pad) width/height
    """
    tile_size: int = 640
    overlap: float = 0.20          # 0.20 ~ 0.25 recommended
    pad_value: int = 114           # or 0 for black, 114 for gray-ish

    title: str = "sliding_tile_pad_640"

    # per input image
    src_hw: List[Tuple[int, int]] = []
    tile_starts_xy: List[List[Tuple[int, int]]] = []  # per image: [(ox,oy), ...]

    # flattened per tile (aligned with output tiles)
    tile_src_idx: List[int] = []
    tile_offset_xy: List[Tuple[int, int]] = []
    tile_valid_hw: List[Tuple[int, int]] = []         # (valid_w, valid_h) before padding
    src_tile_ranges: List[Tuple[int, int]] = []       # per src img: (start_tile_idx, end_tile_idx)

    def model_post_init(self, context):
        self.title = f"sliding_tile_pad_{self.tile_size}"
        return super().model_post_init(context)

    def _make_starts(self, length: int) -> List[int]:
        if length <= self.tile_size:
            return [0]
        stride = int(round(self.tile_size * (1.0 - self.overlap)))
        stride = max(1, stride)

        starts = list(range(0, length - self.tile_size + 1, stride))
        last = length - self.tile_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    def validate_img(self, img_idx, img):
        img.require_np_uint()
        img.require_HW_or_HWC()
        h, w = img.info.H, img.info.W

        # store per-image info
        self.src_hw.append((h, w))

        xs = self._make_starts(w)
        ys = self._make_starts(h)
        coords = [(ox, oy) for oy in ys for ox in xs]
        self.tile_starts_xy.append(coords)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List["ImageMatInfo"] = [],
        meta: Dict[str, Any] = {},
    ) -> List[np.ndarray]:
        # IMPORTANT: reset flattened lists per call (avoid leakage across runs)
        self.tile_src_idx = []
        self.tile_offset_xy = []
        self.tile_valid_hw = []
        self.src_tile_ranges = []

        out_tiles: List[np.ndarray] = []
        tile_cursor = 0

        for src_i, img in enumerate(imgs_data):
            h, w = img.shape[:2]
            coords = self.tile_starts_xy[src_i]

            start_idx = tile_cursor
            for (ox, oy) in coords:
                tile = img[oy:oy + self.tile_size, ox:ox + self.tile_size]
                th, tw = tile.shape[:2]  # valid size before pad

                # pad to 640x640 on right/bottom if needed
                if th != self.tile_size or tw != self.tile_size:
                    if tile.ndim == 2:
                        padded = np.full((self.tile_size, self.tile_size), self.pad_value, dtype=tile.dtype)
                        padded[:th, :tw] = tile
                    else:
                        c = tile.shape[2]
                        padded = np.full((self.tile_size, self.tile_size, c), self.pad_value, dtype=tile.dtype)
                        padded[:th, :tw, :] = tile
                    tile = padded

                out_tiles.append(tile)

                # record flattened meta
                self.tile_src_idx.append(src_i)
                self.tile_offset_xy.append((ox, oy))
                self.tile_valid_hw.append((tw, th))
                tile_cursor += 1

            end_idx = tile_cursor
            self.src_tile_ranges.append((start_idx, end_idx))

        # expose info to downstream postprocess (decode -> clip -> offset -> global NMS)
        meta["tile_info"] = {
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "pad_value": self.pad_value,
            "src_hw": self.src_hw,                       # [(H,W), ...]
            "tile_src_idx": self.tile_src_idx,           # len == num_tiles
            "tile_offset_xy": self.tile_offset_xy,       # len == num_tiles
            "tile_valid_hw": self.tile_valid_hw,         # len == num_tiles (valid_w, valid_h)
            "src_tile_ranges": self.src_tile_ranges,     # per src img: (start,end)
        }

        return out_tiles
```

And here is the **postprocess utility** that you call *after* `decode_boxes` (exactly as you requested): **clip to valid area** (pre-pad), then add offsets, then per-image **Global NMS**.

```python
from typing import Sequence
import torch
from torchvision.ops import batched_nms


def _clip_and_filter_xyxy(boxes: torch.Tensor, valid_w: int, valid_h: int) -> torch.Tensor:
    # boxes: [N,4] xyxy in tile coords (0..640)
    boxes[:, 0] = boxes[:, 0].clamp(0, valid_w)
    boxes[:, 2] = boxes[:, 2].clamp(0, valid_w)
    boxes[:, 1] = boxes[:, 1].clamp(0, valid_h)
    boxes[:, 3] = boxes[:, 3].clamp(0, valid_h)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return keep


@torch.no_grad()
def merge_tiles_global_nms(
    # per-tile decoded outputs (same order as tiles from SlidingTilePad640)
    tile_boxes: Sequence[torch.Tensor],   # each: [Ni,4] xyxy in tile coords
    tile_scores: Sequence[torch.Tensor],  # each: [Ni]
    tile_labels: Sequence[torch.Tensor],  # each: [Ni] long

    tile_info: dict,                      # meta["tile_info"]
    iou_thresh: float = 0.5,
    score_thresh: float = 0.25,
    class_agnostic: bool = False,
    device: Optional[str] = None,
):
    """
    Returns per-source-image lists:
      out_boxes[src_i], out_scores[src_i], out_labels[src_i]
    """
    src_hw = tile_info["src_hw"]
    tile_src_idx = tile_info["tile_src_idx"]
    tile_offset_xy = tile_info["tile_offset_xy"]
    tile_valid_hw = tile_info["tile_valid_hw"]

    num_src = len(src_hw)
    per_boxes = [[] for _ in range(num_src)]
    per_scores = [[] for _ in range(num_src)]
    per_labels = [[] for _ in range(num_src)]

    for t in range(len(tile_boxes)):
        boxes = tile_boxes[t]
        scores = tile_scores[t]
        labels = tile_labels[t]

        if device is not None:
            boxes = boxes.to(device)
            scores = scores.to(device)
            labels = labels.to(device)

        # score filter
        m = scores >= score_thresh
        boxes, scores, labels = boxes[m], scores[m], labels[m]
        if boxes.numel() == 0:
            continue

        # ★ requested: after decode_boxes, clip to ORIGINAL tile valid region (exclude padded area)
        vw, vh = tile_valid_hw[t]  # valid_w, valid_h
        keep = _clip_and_filter_xyxy(boxes, vw, vh)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if boxes.numel() == 0:
            continue

        # offset back to original image coords
        ox, oy = tile_offset_xy[t]
        boxes[:, [0, 2]] += ox
        boxes[:, [1, 3]] += oy

        src_i = tile_src_idx[t]
        per_boxes[src_i].append(boxes)
        per_scores[src_i].append(scores)
        per_labels[src_i].append(labels)

    out_boxes, out_scores, out_labels = [], [], []
    for src_i, (H, W) in enumerate(src_hw):
        if len(per_boxes[src_i]) == 0:
            out_boxes.append(torch.zeros((0, 4), device=device))
            out_scores.append(torch.zeros((0,), device=device))
            out_labels.append(torch.zeros((0,), dtype=torch.long, device=device))
            continue

        boxes = torch.cat(per_boxes[src_i], dim=0)
        scores = torch.cat(per_scores[src_i], dim=0)
        labels = torch.cat(per_labels[src_i], dim=0)

        # optional clamp to image bounds
        boxes[:, 0] = boxes[:, 0].clamp(0, W)
        boxes[:, 2] = boxes[:, 2].clamp(0, W)
        boxes[:, 1] = boxes[:, 1].clamp(0, H)
        boxes[:, 3] = boxes[:, 3].clamp(0, H)

        if class_agnostic:
            keep = torch.ops.torchvision.nms(boxes, scores, iou_thresh)
        else:
            keep = batched_nms(boxes, scores, labels, iou_thresh)

        out_boxes.append(boxes[keep])
        out_scores.append(scores[keep])
        out_labels.append(labels[keep])

    return out_boxes, out_scores, out_labels
```

This gives you the exact 3-step behavior you described, in a style compatible with your `ImageMatProcessor` pattern:

* **tiles padded to 640**
* **overlap sliding**
* **decode → clip to valid (pre-pad) → offset → global NMS**

If you tell me what your `decode_boxes` returns (xyxy? cxcywh? numpy/torch? per-batch format?), I can adjust the `merge_tiles_global_nms()` signature so it plugs in with zero glue code.
