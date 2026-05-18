import json
import os
from pathlib import Path

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_feature(path):
    data = np.load(path, allow_pickle=True)
    if data.shape == ():
        data = data.item()
    return data


def _lookup(features, video_id, index):
    if isinstance(features, dict):
        return features[video_id]
    return features[index]


class MultiModalDataset(Dataset):
    def __init__(self, ep_names, sampling_type=None, **kwargs):
        self.records = list(ep_names)
        self.max_cap = kwargs.get("max_cap", 25)
        self.modality = kwargs.get("modality", "both")
        self.video_features = _load_feature(kwargs["video_feature_path"])
        self.text_features = _load_feature(kwargs["text_feature_path"])

    @classmethod
    def from_bliss(cls, annotation_path, video_feature_path, text_feature_path, max_cap=25, modality="both"):
        with open(annotation_path, "r") as f:
            data = json.load(f)
        records = []
        for feature_key, item in data.items():
            item = dict(item)
            item["feature_key"] = feature_key
            records.append(item)
        return cls(
            records,
            max_cap=max_cap,
            modality=modality,
            video_feature_path=video_feature_path,
            text_feature_path=text_feature_path,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        video_id = item["video_id"]
        feature_key = item.get("feature_key", video_id)
        vid = np.asarray(_lookup(self.video_features, feature_key, idx), dtype=np.float32)
        dia = np.asarray(_lookup(self.text_features, feature_key, idx), dtype=np.float32)

        if vid.ndim == 2:
            vid = vid[:, None, :]
        if dia.ndim == 2:
            dia = dia[:, None, :]

        vid_labels = np.asarray(item["video_label"], dtype=np.float32)[: vid.shape[0]]
        dia_labels = np.asarray(item["text_label"], dtype=np.float32)[: dia.shape[0]]
        vid = vid[: len(vid_labels)]
        dia = dia[: len(dia_labels)]

        return {
            "video_name": feature_key,
            "vid_enc": vid,
            "vid_mask": np.ones(vid.shape[:2], dtype=bool),
            "vid_idx": np.tile(np.arange(vid.shape[1], dtype=np.int64), (vid.shape[0], 1)),
            "vid_labels": vid_labels,
            "dia_enc": dia,
            "word_mask": np.ones(dia.shape[:2], dtype=bool),
            "dia_mask": np.ones(dia.shape[0], dtype=bool),
            "dia_labels": dia_labels,
        }

    def collate_fn(self, batch):
        bsz = len(batch)
        max_vid = max(x["vid_enc"].shape[0] for x in batch)
        max_vcap = max(x["vid_enc"].shape[1] for x in batch)
        vid_dim = batch[0]["vid_enc"].shape[-1]
        max_dia = max(x["dia_enc"].shape[0] for x in batch)
        max_dcap = max(x["dia_enc"].shape[1] for x in batch)
        dia_dim = batch[0]["dia_enc"].shape[-1]
        seq_len = max_vid + max_dia

        vid_enc = np.zeros((bsz, max_vid, max_vcap, vid_dim), dtype=np.float32)
        vid_mask = np.zeros((bsz, max_vid, max_vcap), dtype=bool)
        vid_idx = np.zeros((bsz, max_vid, max_vcap), dtype=np.int64)
        vid_labels = np.zeros((bsz, max_vid), dtype=np.float32)
        dia_enc = np.zeros((bsz, max_dia, max_dcap, dia_dim), dtype=np.float32)
        word_mask = np.zeros((bsz, max_dia, max_dcap), dtype=bool)
        dia_mask = np.zeros((bsz, max_dia), dtype=bool)
        dia_labels = np.zeros((bsz, max_dia), dtype=np.float32)
        bin_indices = np.zeros((bsz, seq_len), dtype=np.int64)
        token_type = np.full((bsz, seq_len), 3, dtype=np.int64)
        mask = np.zeros((bsz, seq_len), dtype=bool)
        group_idx = np.zeros((bsz, seq_len), dtype=np.int64)
        subgroup_len = np.zeros((bsz, 1, 2), dtype=np.int64)
        video_names = []

        for i, item in enumerate(batch):
            v, vc = item["vid_enc"].shape[:2]
            d, dc = item["dia_enc"].shape[:2]
            video_names.append(item["video_name"])
            vid_enc[i, :v, :vc] = item["vid_enc"]
            vid_mask[i, :v, :vc] = item["vid_mask"]
            vid_idx[i, :v, :vc] = item["vid_idx"]
            vid_labels[i, :v] = item["vid_labels"]
            dia_enc[i, :d, :dc] = item["dia_enc"]
            word_mask[i, :d, :dc] = item["word_mask"]
            dia_mask[i, :d] = item["dia_mask"]
            dia_labels[i, :d] = item["dia_labels"]
            bin_indices[i, :v] = np.arange(v)
            bin_indices[i, max_vid : max_vid + d] = np.arange(d)
            token_type[i, :v] = 0
            token_type[i, max_vid : max_vid + d] = 1
            mask[i, :v] = True
            mask[i, max_vid : max_vid + d] = True
            subgroup_len[i, 0] = [v, d]

        feat_dict = (
            {
                "vid_enc": torch.from_numpy(vid_enc),
                "vid_mask": torch.from_numpy(vid_mask),
                "vid_idx": torch.from_numpy(vid_idx),
                "labels": torch.from_numpy(vid_labels),
            },
            {
                "dia_enc": torch.from_numpy(dia_enc),
                "word_mask": torch.from_numpy(word_mask),
                "dia_mask": torch.from_numpy(dia_mask),
                "labels": torch.from_numpy(dia_labels),
            },
        )
        labels = torch.zeros(bsz)
        return (
            feat_dict,
            video_names,
            torch.from_numpy(bin_indices),
            torch.from_numpy(token_type),
            torch.from_numpy(mask),
            torch.from_numpy(group_idx),
            torch.from_numpy(subgroup_len),
            labels,
        )


def load_edge_dict(cfg=None):
    graph_cfg = getattr(cfg, "graph", {}) if cfg is not None else {}
    return {
        "zeta_v": float(graph_cfg.get("zeta_v", 0.95)),
        "zeta_t": float(graph_cfg.get("zeta_t", 0.97)),
        "shot_constraint": bool(graph_cfg.get("shot_constraint", True)),
    }


def load_alignment_reference():
    return None


def _cosine_edges(features, threshold, relation, shot_constraint=False):
    num_nodes = features.shape[0]
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)
    norm = torch.nn.functional.normalize(features.detach().float().cpu(), dim=-1)
    sim = norm @ norm.T
    mask = sim >= threshold
    mask.fill_diagonal_(True)
    if shot_constraint and num_nodes > 1:
        idx = torch.arange(num_nodes)
        adjacent = (idx[:, None] - idx[None, :]).abs() == 1
        mask = mask & ~adjacent
        mask.fill_diagonal_(True)
    src, dst = torch.nonzero(mask, as_tuple=True)
    return src.to(torch.int64), dst.to(torch.int64)


def _temporal_alignment(num_dia, num_vid):
    if num_dia == 0 or num_vid == 0:
        return torch.zeros((num_dia, num_vid), dtype=torch.bool)
    align = torch.zeros((num_dia, num_vid), dtype=torch.bool)
    for dia_idx in range(num_dia):
        center = int(round(dia_idx * max(num_vid - 1, 0) / max(num_dia - 1, 1)))
        lo = max(0, center - 1)
        hi = min(num_vid, center + 2)
        align[dia_idx, lo:hi] = True
    return align


def load_graph(video_name, edge_dict, graph_label):
    num_dia = int(graph_label["dia"].numel())
    num_vid = int(graph_label["video"].numel())
    dia_src, dia_dst = _cosine_edges(graph_label["dia_feat"], edge_dict["zeta_t"], "dia_sim")
    vid_src, vid_dst = _cosine_edges(graph_label["video_feat"], edge_dict["zeta_v"], "video_sim", edge_dict["shot_constraint"])
    align = _temporal_alignment(num_dia, num_vid)
    align_src, align_dst = torch.nonzero(align, as_tuple=True)
    graph = dgl.heterograph(
        {
            ("dia", "dia_sim", "dia"): (dia_src, dia_dst),
            ("video", "video_sim", "video"): (vid_src, vid_dst),
            ("dia", "time_align", "video"): (align_src, align_dst),
            ("video", "time_align_by", "dia"): (align_dst, align_src),
        },
        num_nodes_dict={"dia": num_dia, "video": num_vid},
    )
    possible = max(num_dia * num_dia + num_vid * num_vid + 2 * num_dia * num_vid, 1)
    actual = int(dia_src.numel() + vid_src.numel() + 2 * align_src.numel())
    return graph, align, actual / possible
