# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            batch = next(self.loader)
            self.next_samples, self.next_targets = batch[0], batch[1]
            self.next_sketches = batch[2] if len(batch) > 2 else None
            self.next_cat_embeds = batch[3] if len(batch) > 3 else None
            self.next_cat_ids = batch[4] if len(batch) > 4 else None
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            self.next_sketches = None
            self.next_cat_embeds = None
            self.next_cat_ids = None
            return
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            if self.next_sketches is not None:
                self.next_sketches = tuple(
                    s.to(self.device, non_blocking=True) for s in self.next_sketches
                )
            if self.next_cat_embeds is not None:
                self.next_cat_embeds = tuple(
                    e.to(self.device, non_blocking=True) if isinstance(e, torch.Tensor) else e
                    for e in self.next_cat_embeds
                )

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            sketches = self.next_sketches
            cat_embeds = self.next_cat_embeds
            cat_ids = self.next_cat_ids
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                batch = next(self.loader)
                samples, targets = batch[0], batch[1]
                sketches = batch[2] if len(batch) > 2 else None
                cat_embeds = batch[3] if len(batch) > 3 else None
                cat_ids = batch[4] if len(batch) > 4 else None
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = targets = sketches = cat_embeds = cat_ids = None
        return samples, targets, sketches, cat_embeds, cat_ids
