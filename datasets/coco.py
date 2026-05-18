# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, # Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import bisect
from pathlib import Path
import pdb
import time
import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import random
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size, get_rank, is_dist_avail_and_initialized
import torch.distributed as dist
import torchvision
from torchvision.transforms import Normalize, Compose, Resize, InterpolationMode, CenterCrop, ToTensor
from numpy.random import choice
from pycocotools.coco import COCO
import pickle as pkl
from typing import Any, Dict, List, Tuple, Sequence

import datasets.transforms as T
from tqdm import tqdm
import pickle
from collections import defaultdict
import random
import numpy as np
from PIL import Image, ImageDraw
import os
import json
from torchvision.utils import save_image


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def rasterize_stroke3(strokes: np.ndarray, size: int = 224,
                      line_width: int = 2, padding: int = 10) -> Image.Image:
    """
    Render a stroke-3 array (dx, dy, pen_state) to a white-background PIL image.
    pen_state 0 = pen down, 1 = pen up (end of stroke).
    Ported from SLIP/datasets.py.
    """
    abs_coords = np.cumsum(strokes[:, :2], axis=0).astype(float)
    pen_states = strokes[:, 2]
    x, y = abs_coords[:, 0], abs_coords[:, 1]
    x_range = x.max() - x.min() or 1
    y_range = y.max() - y.min() or 1
    scale = (size - 2 * padding) / max(x_range, y_range)
    x = ((x - x.min()) * scale + padding).astype(int)
    y = ((y - y.min()) * scale + padding).astype(int)
    img = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    stroke_pts: list = []
    for i in range(len(strokes)):
        stroke_pts.append((int(x[i]), int(y[i])))
        if pen_states[i] == 1:
            if len(stroke_pts) >= 2:
                draw.line(stroke_pts, fill=(0, 0, 0), width=line_width)
            stroke_pts = []
    if len(stroke_pts) >= 2:
        draw.line(stroke_pts, fill=(0, 0, 0), width=line_width)
    return img


def normalize_transform():
    return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

class CocoDetectionSketch(torchvision.datasets.CocoDetection): # Closed/Open-world settings, wrapper around CocoDetection
    """
    Template Dataset for COCO + sketch datasets:
        - Subclasses MUST provide:
            1) `all_categories` (abstract property returning a Sequence[str])
            2) `_setup_sketches` (abstract method that fills `self.cls2sketch`)
            3) `_process_sketch` (abstract method that processes a sketch file and returns a tensor)
        - Subclasses MAY override:
            1) `_sanity_check` (method that checks that the provided args are correct)
            2) `_save_seen_unseen_split` (method that returns a list of unseen categories)
            3) `_construct_coco_subset` (method that returns the annotations and image ids for the selected categories)
    """
    # --------- REQUIRED CONTRACT ---------
    @property
    @abstractmethod
    def sketch_name(self) -> str:
        """Short name of the sketch dataset (e.g., 'qd', 'sk')."""
        raise NotImplementedError
    @property
    @abstractmethod
    def all_categories(self) -> Sequence[str]:
        """ List of all categories names (e.g., the 56 COCO↔QD overlap)."""
        raise NotImplementedError
    
    @abstractmethod
    def _setup_sketches(self) -> None:
        """
            Populate `self.cls2sketch: Dict[str, List[str]]` with paths to sketch files.
            Do NOT take a sketch path in __init__; plug it in here.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _process_sketch(self, sketch_path:str) -> torch.Tensor:
        """Process the sketch file and return a tensor representation.

        Args:
            sketch_path (str): Path to the sketch file.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.

        Returns:
            torch.Tensor: Processed sketch as a tensor.
        """
        raise NotImplementedError
    
    # --------- BASE IMPLEMENTATION ---------
    def __init__(
            self,
            image_set:str,
            img_folder:str|Path,
            ann_file:str,
            transforms,
            return_masks:bool,
            unroll_all_cats: bool = False, # Use to make eval deterministic
            data_frac:float = 1.0,
            train_scheme_world:str = "open",
            ds_len:int = 0,
            multi_sketch:int = 0, # If >1, uses multiple sketches per class
            seed:int = 42,
            inference:bool = False,
            sketch_root: str | None = None,  # root dir for sketch data (subclass-specific)
        ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.image_set = image_set
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.unroll_all_cats = unroll_all_cats
        self._transforms = transforms
        self.return_masks = return_masks
        self.data_frac = data_frac
        self.train_scheme_world = train_scheme_world
        self.ds_len = ds_len
        self.multi_sketch = multi_sketch
        self.inference = inference
        self.sketch_root = sketch_root

        with open(ann_file) as f: 
            json_file = json.load(f)
        with open('./checkpoints/clip_model/text_embeddings.pkl', 'rb') as f:
            self.text_embeds = pickle.load(f)
        ROOT, _ = os.path.split(img_folder) # Picks the path/to/annots/ from path/to/annots/train
        self.coco_root = ROOT
        
        self.cls2sketch: Dict[str, List[str]] = {}
        self.id2class: Dict[int, str] = {}
        self.class2id: Dict[str, int] = {}
        
        for cat in json_file['categories']:
            self.id2class[cat['id']] = cat['name']
            self.class2id[cat['name']] = cat['id']
        
        self._sanity_check()
        unseen_cats = self._save_seen_unseen_split()
        annotate, seen_image_ids = self._construct_coco_subset(unseen_cats, json_file)
        images = []
        for image in tqdm(json_file['images']):
            img_id = image['id']
            if img_id in seen_image_ids: # If in the selected pool of images
                images.append(image)
        
        # Rebuild the json file for seen/unseen split
        json_file['annotations'] = annotate
        json_file['images'] = images
        # FINAL coco split stats
        print(f"[FINAL] annot_len: {len(annotate)}, img_len: {len(images)}")
        
        # Save the modified json file
        temp_ann_file = os.path.join(f'annotations/temp_{self.sketch_name}_{self.image_set}_{self.train_scheme_world}.json')
        os.makedirs(os.path.dirname(temp_ann_file), exist_ok=True) # Create the directory if it does not exist
        if get_rank() == 0:
            with open(temp_ann_file, 'w') as f:
                json.dump(json_file, f)
        if is_dist_avail_and_initialized():
            dist.barrier() # WAIT: IO operations above
        super().__init__(img_folder, temp_ann_file) # Use the default coco class on modified open-set/ closed-set json file
        self.prepare = ConvertCocoPolysToMask(return_masks)

        self.transforms_sketch = Compose([
            ToTensor(),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                      std=(0.26862954, 0.26130258, 0.27577711)) # Values taken from CLIP preprocess
            ])
        self._setup_sketches()
        # build the (img, cat) index if we want deterministic unrolling
        if self.unroll_all_cats:
            self._build_unrolled_index()
        # if self.image_set == 'train': # Reseeding random for train cat selection, not for class-split
        #     t = 1000 * time.time()
        #     random.seed(t)
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)
        
    def __getitem__(self, idx):
        if self.unroll_all_cats:
            image_id, selected_cat = self.unrolled_index[idx] # lookup
            ds_idx = self.imgid2dsidx[image_id] # get the dataset idx
            img, target = super().__getitem__(ds_idx)
        else:
            img, target = super().__getitem__(idx)
            image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        categories = list(set(target['labels'].tolist()))
        new_target = {}
        if not self.unroll_all_cats: # already assigned above if unrolling
            selected_cat = random.choice(categories)
        
        keep = target['labels']==selected_cat
        
        selected_keys = ['boxes', 'labels', 'area', 'iscrowd', 'masks']
        for key, value in target.items():
            if key in selected_keys:
                new_target[key] = value[keep]
            else:
                new_target[key] = value
        # make it class-agnostic for the detector
        new_target['labels'] = torch.ones_like(new_target['labels'])
        selected_cat = self.id2class[selected_cat]
        selected_cat_embed = self.text_embeds[selected_cat]
        if self.image_set == 'val':
            selected_cat_embed = torch.zeros_like(self.text_embeds[selected_cat])
        
        # grab sketches for that class
        sketch_paths = random.choices(self.cls2sketch[selected_cat], k=self.multi_sketch if self.multi_sketch else 1)
        sketch_list = []
        
        for spath in sketch_paths:
            sketch = self._process_sketch(spath)
            sketch_list.append(sketch.unsqueeze(0))
        sketch_list = torch.cat(sketch_list, dim=0)
        old_boxes = new_target['boxes'].clone()
        if self._transforms is not None:
            img, new_target = self._transforms(img, new_target)

        cat_id = -1
        if self.image_set == 'train':
            cat_id = self.class2id[selected_cat]
        else:
            new_target['boxes'] = old_boxes

        if self.inference: # Prints img_path and sketch_path for debugging
            img_info = self.coco.loadImgs(image_id)[0]
            img_relpath = img_info['file_name']
            img_fullpath = os.path.join(self.img_folder, img_relpath)
            print(f"[DATA INFERENCE] img_path: {img_fullpath}, sketch_path: {sketch_paths}, cat: {selected_cat}")
            # DEBUG: Hacky pass for inference viz
            selected_cat_embed = (img_fullpath, sketch_paths, selected_cat)
        return img, new_target, sketch_list, selected_cat_embed, cat_id
    
    def _build_unrolled_index(self):
        """
        Build a flat list of (img_id, cat_id) so that __getitem__(i)
        always refers to exactly one category in exactly one image.
        """
        self.unrolled_index = []
        # map coco image_id → dataset index (the index torchvision uses)
        self.imgid2dsidx = {img_id: ds_idx for ds_idx, img_id in enumerate(self.ids)}
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            # unique category ids present in this image
            cat_ids = {ann["category_id"] for ann in anns}
            for cat_id in sorted(cat_ids):
                self.unrolled_index.append((img_id, cat_id))
        print(f"[UNROLL] built {len(self.unrolled_index)} (image,category) pairs")
        
    def __len__(self):
        if self.ds_len > 0:
            return self.ds_len
        elif self.ds_len == 0:
            if self.unroll_all_cats:
                return len(self.unrolled_index) # len() = #image-cat pairs
            return super().__len__()
        else:
            raise ValueError(f"Dataset length {self.ds_len} is invalid.")
    def _sanity_check(self):
        # Sanity Checks
        if self.image_set == 'train':
            if self.train_scheme_world == "closed":
                print("TRAINING CLOSED SET STATS")
            elif self.train_scheme_world == "open":
                print("TRAINING OPEN SET STATS")
            else:
                raise ValueError(f"Unknown training scheme: {self.train_scheme_world}")
        elif self.image_set == 'val':
            print("VALIDATION SET STATS")
        else:
            raise ValueError(f"Unknown image set: {self.image_set}")
        print(f"Image folder: {self.img_folder}, annot file: {self.ann_file}")
        cats = self.all_categories
        if not isinstance(cats, Sequence) or len(cats) == 0:
            raise ValueError("Subclass must provide a non-empty `all_categories` sequence.")
    def _save_seen_unseen_split(self):
        unseen_cats = []
        if self.train_scheme_world == "open":
            n_holdout = len(self.all_categories) // 4 # Leave out 25% categories
            unseen_cats = random.sample(self.all_categories, n_holdout)
            if get_rank() == 0:
                seen_cats = list(set(self.all_categories) - set(unseen_cats))
                with open(f"outputs/unseen_cats_coco-{self.sketch_name}.txt", "w+") as f: # For debugging purposes
                    f.write("\n".join(unseen_cats))
                with open(f"outputs/seen_cats_coco-{self.sketch_name}.txt", "w+") as f: # For debugging purposes
                    f.write("\n".join(seen_cats))
                del seen_cats # Only needed for logs
        if get_rank() == 0:
            with open(f"outputs/common_cats_coco-{self.sketch_name}.txt", "w+") as f:
                f.write("\n".join(self.all_categories))
            print(f"Saving categories to 'outputs/'")
        if is_dist_avail_and_initialized():
            dist.barrier() # WAIT: IO operations above
        return unseen_cats
    def _construct_coco_subset(self, unseen_cats:dict=None, json_file:dict=None):
        seen_image_ids = {} # becomes whole set for val in closed world
        unseen_image_ids = {}
        # Build annotate that has the selected categories' images only, with no overlap between seen and unseen categories' images
        annotate = []
        for anno in json_file['annotations']:
            cat_id = anno['category_id']
            img_id = anno['image_id']
            
            if self.id2class[cat_id] in self.all_categories: # Select only in the 56 common categories
                if self.image_set == 'train' or self.train_scheme_world == "closed": # (train) or (closed-val)
                    if self.id2class[cat_id] not in unseen_cats: # Select only the seen categories out of the 56 common categories from train set
                        annotate.append(anno)
                        seen_image_ids.setdefault(cat_id, []).append(img_id) 
                    else: unseen_image_ids.setdefault(cat_id, []).append(img_id) # Keep track of unseen categories
                else: # val
                    if self.id2class[cat_id] in unseen_cats: # Select only the unseen categories out of the 56 common categories from val set
                        annotate.append(anno)
                        seen_image_ids.setdefault(cat_id, []).append(img_id) 
        
        print(f"[TOTAL] Set of selected categories: {len(seen_image_ids)}, withheld categories: {len(unseen_image_ids)}")
        print(f"[TOTAL] Set of selected images per category: {[f'{key}:{len(val)}' for key,val in seen_image_ids.items()]}")
        
        if self.image_set == 'train': # Only need to create a subset of the training set
            print(f"[SUBSET] : data_frac: {self.data_frac}")
            subset_seen_image_ids =  {key: set(choice(val, int(self.data_frac * len(val))).tolist()) for key,val in seen_image_ids.items()}
            subset_unseen_image_ids =  {key: set(choice(val, int(self.data_frac * len(val))).tolist()) for key,val in unseen_image_ids.items()}
            seen_image_ids = subset_seen_image_ids
            unseen_image_ids = subset_unseen_image_ids
            print(f"[SUBSET] Set of selected categories: {len(seen_image_ids)}, withheld categories: {len(unseen_image_ids)}")
            print(f"[SUBSET] Set of selected images per category: {[f'{key}:{len(val)}' for key,val in seen_image_ids.items()]}")
        
        # convert to list of sets to final set
        seen_image_ids = set().union(*seen_image_ids.values())
        unseen_image_ids = set().union(*unseen_image_ids.values())
        seen_image_ids -= unseen_image_ids # Remove the unseen categories imgs from the seen categories imgs (prevent data leakage)
        print(f"[FINAL] set_selected_len: {len(seen_image_ids)}")
        return annotate, seen_image_ids
    
class CocoDetectionQD(CocoDetectionSketch):
    _ALL = ['bicycle', 'car', 'airplane', 'bus', 'train', 'truck', 'traffic light', 'fire hydrant', 'stop sign', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'suitcase', 'baseball bat', 'skateboard', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'banana', 'apple', 'sandwich', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'bed', 'toilet', 'laptop', 'mouse', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'book', 'clock', 'vase', 'scissors', 'toothbrush'] # All the common categories with QD!
    _NAME = 'qd'
    @property
    def sketch_name(self) -> str:
        return self._NAME
    @property
    def all_categories(self) -> Sequence[str]:
        return self._ALL

    def _setup_sketches(self) -> None:
        """
        Load QuickDraw SketchRNN data for all common categories using memory-mapped
        ptr/strokes npy files (format from SLIP/datasets.py).

        `self.sketch_root` must point to a directory containing
        `{class_name}.{split}.ptr.npy` and `{class_name}.{split}.strokes.npy`.

        `self.cls2sketch[cat]` is populated with `"{cat}:{sample_idx}"` strings so
        that the base-class random.choices + _process_sketch loop works unchanged.
        """
        assert self.sketch_root is not None, \
            "CocoDetectionQD requires sketch_root to point to the SketchRNN npy directory."
        split = 'train' if self.image_set == 'train' else 'valid'
        print(f"Loading Quick,Draw! ({split}) from {self.sketch_root} ...")
        # _qd_mmap: cat -> (ptr_array, strokes_mmap) — shared across workers via OS page cache
        self._qd_mmap: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for cat in self.all_categories:
            ptr = np.load(os.path.join(self.sketch_root, f'{cat}.{split}.ptr.npy'))
            strokes = np.load(os.path.join(self.sketch_root, f'{cat}.{split}.strokes.npy'),
                              mmap_mode='r')
            self._qd_mmap[cat] = (ptr, strokes)
            n = len(ptr) - 1
            self.cls2sketch[cat] = [f'{cat}:{i}' for i in range(n)]

    def _process_sketch(self, sketch_ref: str) -> torch.Tensor:
        """Decode a "{cat}:{idx}" ref, rasterise the stroke-3 array, return a tensor."""
        cat, idx_str = sketch_ref.rsplit(':', 1)
        idx = int(idx_str)
        ptr, strokes = self._qd_mmap[cat]
        stroke_data = strokes[ptr[idx]:ptr[idx + 1]]
        img = rasterize_stroke3(stroke_data)
        return self.transforms_sketch(img)
    

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

class CocoDetectionSketchy(CocoDetectionSketch): # Closed/Open-world settings, Sketchy wrapper around CocoDetection
    _ALL = ['elephant', 'bear', 'cat', 'zebra',  'horse', 'giraffe', 'airplane', 'dog', 'scissors', 'pizza', 'cow','umbrella', 'sheep', 'bicycle', 'hot dog', 'banana', 'couch','bench', 'chair', 'apple','cup', 'car', 'knife', 'clock', 'spoon', 'mouse']
    _NAME = 'sk'
    @property
    def sketch_name(self) -> str:
        return self._NAME

    @property
    def all_categories(self) -> Sequence[str]:
        # Dataset specific: Sketchy
        return self._ALL

    def _setup_sketches(self)-> None:
        """
        Reads sketch paths and populates `self.class2sketch[cat] = [list of files]`.
        NOTE: No argument—path is chosen here, not passed into __init__.
        """
        print("Loading Sketchy ...")
        sketchy_root = Path(self.coco_root) / "sketch_data"
        _sketchy_path = os.path.join(sketchy_root,"sketchy_dataset.pkl")
        with open(_sketchy_path, 'rb') as f:
            _sketchy_path = pickle.load(f)
        split_key = 'train' if self.image_set == 'train' else 'test'
        
        for cat in _sketchy_path[split_key]:                
            for img_file in _sketchy_path[split_key][cat]:
                path = os.path.join(sketchy_root, "images", f"{img_file}.png")
                val=self.cls2sketch.get(cat,[])
                val.append(path)
                self.cls2sketch[cat]=val # Store class wise Sketchy paths
        # Check if the categories exist on Sketchy
        for cat in self.all_categories:
            # pdb.set_trace()
            if len(self.cls2sketch[cat]) == 0:
                self.all_categories.remove(cat)
                print(f"WARNING: No sketches found for category: {cat}. Removing from common categories list.")
        
    def _process_sketch(self, sketch_path:str) -> torch.Tensor:
        sketch = Image.open(sketch_path).convert('RGB')
        sketch = 255 - np.array(sketch)
        sketch = Image.fromarray(sketch)
        return self.transforms_sketch(sketch)

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
        
    if args.eval_type=='coco-sketchy':
        DatasetClass = CocoDetectionSketchy
    elif args.eval_type=='coco-qd':
        DatasetClass = CocoDetectionQD
    else:
        raise ValueError(f"Unknown eval type: {args.eval_type}")

    img_folder, ann_file = PATHS[image_set]

    dataset = DatasetClass(
                image_set,
                img_folder,
                ann_file,
                transforms=make_coco_transforms(image_set),
                return_masks=args.masks,
                unroll_all_cats=True if image_set=='val' else False, # change eval for deterministic class selection
                data_frac=args.data_frac,
                train_scheme_world=args.train_scheme_world,
                ds_len=args.debug_size,
                multi_sketch=args.multi_sketch,
                seed=args.seed,
                inference=args.inference,
                sketch_root=getattr(args, 'sketch_root', None),
            )
    return dataset