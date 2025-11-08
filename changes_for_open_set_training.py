import argparse
import json
import pdb
from datasets.coco import build

# def create_argparse_stub():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--coco_path', type=str, default='/home/rahul/coco')
#     parser.add_argument('--masks',type=bool, default=False)
#     parser.add_argument('--cache_mode', type=bool, default=False)
#     return parser.parse_args()

split=['train', 'val'] # train or val
for s in split:
    split = s
    ann_file = f'/home/rahul/coco/annotations/instances_{split}2017.json'
    annotate = []
    unseen_cats = set() # set of unseen category names
    with open('outputs/unseen_cats_qd_sk.txt') as f:
        lines = f.readlines()
        unseen_cats = set([line.strip() for line in lines])
    print("Unseen categories:", unseen_cats)
    with open(ann_file) as f: # main json to select from
        json_file = json.load(f)
    class2id = {} # Build class name to id mapping
    for cat in json_file['categories']:
        class2id[cat['name']] = cat['id']
    unseen_cat_ids = set() # set of unseen category ids
    for cat in unseen_cats:
        unseen_cat_ids.add(class2id[cat])

    for annot in json_file['annotations']:
        if annot['category_id'] not in unseen_cat_ids:
            annotate.append(annot)
    selected_cats = []
    for cat_dict in json_file['categories']:
        if cat_dict['id'] not in unseen_cat_ids:
            selected_cats.append(cat_dict)
    # Replace with subset of categories
    json_file['categories'] = selected_cats
    json_file['annotations'] = annotate

    with open(f'/home/rahul/Deformable-DETR/annotations/instances_openset_{split}2017.json', 'w') as f:
        json.dump(json_file, f)