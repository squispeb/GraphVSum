from pathlib import Path

import dgl
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_edge_dict():
    return {}


def load_alignment_reference():
    return None


def _cosine_edges(features, threshold, shot_ids=None, shot_constraint=False,
                  target_density=None):
    num_nodes = int(features.shape[0])
    if num_nodes == 0:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty, 0.0

    feats = torch.as_tensor(features, dtype=torch.float32)
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    sim = feats @ feats.t()
    eye = torch.eye(num_nodes, dtype=torch.bool)
    candidate = sim >= float(threshold)

    if shot_constraint and shot_ids is not None:
        shot_ids = torch.as_tensor(shot_ids, dtype=torch.long)
        same_shot = shot_ids[:, None] == shot_ids[None, :]
        candidate = candidate & (~same_shot | eye)

    candidate = candidate | eye
    off_diag = candidate & ~eye
    if target_density is not None and num_nodes > 1:
        max_edges = int(round(float(target_density) * num_nodes * (num_nodes - 1)))
        edge_count = int(off_diag.sum().item())
        if 0 < max_edges < edge_count:
            scores = sim.masked_fill(~off_diag, -2.0).flatten()
            keep_idx = torch.topk(scores, k=max_edges).indices
            pruned = torch.zeros(num_nodes * num_nodes, dtype=torch.bool)
            pruned[keep_idx] = True
            off_diag = pruned.view(num_nodes, num_nodes)
            candidate = off_diag | eye

    src, dst = torch.nonzero(candidate, as_tuple=True)
    density = float(off_diag.float().mean().item()) if num_nodes > 1 else 0.0
    return src.to(torch.int64), dst.to(torch.int64), density


def load_graph(video_name, edge_dict, graph_label, zeta_v=0.95, zeta_t=0.97,
               shot_constraint=False, target_graph_density=None):
    num_dia = int(graph_label["dia"].shape[0])
    num_video = int(graph_label["video"].shape[0])

    dia_idx = torch.arange(num_dia, dtype=torch.int64)
    if "dia_features" in graph_label and num_dia:
        dia_src, dia_dst, dia_density = _cosine_edges(
            graph_label["dia_features"], zeta_t, target_density=target_graph_density
        )
    else:
        dia_src = dia_dst = dia_idx
        dia_density = 0.0

    if "video_features" in graph_label and num_video:
        video_src, video_dst, video_density = _cosine_edges(
            graph_label["video_features"], zeta_v,
            shot_ids=graph_label.get("shot_ids"),
            shot_constraint=shot_constraint,
            target_density=target_graph_density,
        )
    else:
        video_idx = torch.arange(num_video, dtype=torch.int64)
        video_src = video_dst = video_idx
        video_density = 0.0

    if num_dia and num_video:
        aligned_video = torch.clamp((dia_idx.float() * num_video / num_dia).long(), max=num_video - 1)
        aligned_dia = dia_idx
    else:
        aligned_dia = torch.zeros(0, dtype=torch.int64)
        aligned_video = torch.zeros(0, dtype=torch.int64)

    graph_data = {
        ("dia", "dia_sim", "dia"): (dia_src, dia_dst),
        ("video", "video_sim", "video"): (video_src, video_dst),
        ("dia", "time_align", "video"): (aligned_dia, aligned_video),
        ("video", "time_align_by", "dia"): (aligned_video, aligned_dia),
    }
    graph = dgl.heterograph(graph_data, num_nodes_dict={"dia": num_dia, "video": num_video})

    dv_time_align = torch.zeros((num_dia, num_video), dtype=torch.bool)
    if num_dia and num_video:
        dv_time_align[aligned_dia, aligned_video] = True
    if num_dia and num_video:
        align_density = float(dv_time_align.float().mean().item())
        density = float(np.mean([video_density, dia_density, align_density]))
    else:
        density = video_density if num_video else dia_density
    return graph, dv_time_align, density


class MultiModalDataset(Dataset):
    def __init__(self, ep_names, sampling_type, **kwargs):
        self.data_path = Path(kwargs.get("data_path", "data")) / "TVSum"
        self.modality = kwargs.get("modality", "both")
        self.withGROUP = kwargs.get("withGROUP", True)
        self.limit_samples = kwargs.get("limit_samples")

        self.h5_path = self.data_path / "feature" / "eccv16_dataset_tvsum_google_pool5.h5"
        text_path = self.data_path / "feature" / "text_roberta.npy"
        self.text_features = np.load(text_path, allow_pickle=True).item()

        keys = [str(Path(str(name)).name) for name in ep_names]
        if not keys:
            with h5py.File(self.h5_path, "r") as h5:
                keys = sorted(h5.keys())
        self.samples = [key for key in keys if key in self.text_features]
        if self.limit_samples:
            self.samples = self.samples[: int(self.limit_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key = self.samples[idx]
        with h5py.File(self.h5_path, "r") as h5:
            group = h5[key]
            video = np.asarray(group["features"][:], dtype=np.float32)
            video_labels = np.asarray(group["gtsummary"][:], dtype=np.float32)
            change_points = np.asarray(group["change_points"][:], dtype=np.int64)
            n_frame_per_seg = np.asarray(group["n_frame_per_seg"][:], dtype=np.int64)
            picks = np.asarray(group["picks"][:], dtype=np.int64)
            user_summary = np.asarray(group["user_summary"][:], dtype=np.float32)
            n_frames = int(np.asarray(group["n_frames"][()]))

        text = np.asarray(self.text_features[key], dtype=np.float32)
        video_len = min(video.shape[0], video_labels.shape[0])
        video = video[:video_len]
        video_labels = video_labels[:video_len]
        picks = picks[:video_len]
        shot_ids = np.searchsorted(change_points[:, 1], picks, side="left")
        shot_ids = np.clip(shot_ids, 0, max(len(change_points) - 1, 0))

        text_len = text.shape[0]
        if text_len:
            text_positions = np.linspace(0, max(video_len - 1, 0), text_len).round().astype(int)
            text_labels = video_labels[text_positions]
        else:
            text_labels = np.zeros(0, dtype=np.float32)

        video = torch.from_numpy(video).unsqueeze(1)
        text = torch.from_numpy(text).unsqueeze(1)
        video_labels = torch.from_numpy(video_labels)
        text_labels = torch.from_numpy(text_labels.astype(np.float32))

        feat_dict = (
            {
                "vid_enc": video,
                "vid_mask": torch.ones(video_len, 1),
                "vid_idx": torch.arange(video_len).unsqueeze(1),
                "labels": video_labels,
                "shot_ids": torch.from_numpy(shot_ids.astype(np.int64)),
                "change_points": torch.from_numpy(change_points),
                "n_frame_per_seg": torch.from_numpy(n_frame_per_seg),
                "picks": torch.from_numpy(picks.astype(np.int64)),
                "user_summary": torch.from_numpy(user_summary),
                "n_frames": torch.tensor(n_frames, dtype=torch.long),
            },
            {
                "dia_enc": text,
                "dia_mask": torch.ones(text_len),
                "word_mask": torch.ones(text_len, 1),
                "labels": text_labels,
            },
        )

        bin_indices = torch.arange(video_len + text_len)
        token_type = torch.cat([torch.zeros(video_len), torch.ones(text_len)]).long()
        mask = torch.ones(video_len + text_len)
        group_idx = torch.zeros(video_len + text_len).long()
        subgroup_len = torch.tensor([video_len, text_len]).long()
        labels = torch.cat([video_labels, text_labels])
        return feat_dict, key, bin_indices, token_type, mask, group_idx, subgroup_len, labels

    def collate_fn(self, batch):
        def pad_1d(items, value=0):
            max_len = max(x.shape[0] for x in items)
            out = torch.full((len(items), max_len), value, dtype=items[0].dtype)
            for i, item in enumerate(items):
                out[i, : item.shape[0]] = item
            return out

        def pad_nd(items, value=0):
            max_len = max(x.shape[0] for x in items)
            shape = (len(items), max_len) + tuple(items[0].shape[1:])
            out = torch.full(shape, value, dtype=items[0].dtype)
            for i, item in enumerate(items):
                out[i, : item.shape[0]] = item
            return out

        def pad_user_summary(items, value=0):
            max_users = max(x.shape[0] for x in items)
            max_frames = max(x.shape[1] for x in items)
            out = torch.full((len(items), max_users, max_frames), value, dtype=items[0].dtype)
            for i, item in enumerate(items):
                out[i, : item.shape[0], : item.shape[1]] = item
            return out

        video_dicts, text_dicts = [], []
        names, bin_indices, token_types, masks, group_idxs, subgroup_lens, labels = [], [], [], [], [], [], []
        for feat_dict, name, bin_idx, token_type, mask, group_idx, subgroup_len, label in batch:
            video_dict, text_dict = feat_dict
            video_dicts.append(video_dict)
            text_dicts.append(text_dict)
            names.append(name)
            bin_indices.append(bin_idx)
            token_types.append(token_type)
            masks.append(mask)
            group_idxs.append(group_idx)
            subgroup_lens.append(subgroup_len)
            labels.append(label)

        video_batch = {
            "vid_enc": pad_nd([x["vid_enc"] for x in video_dicts]),
            "vid_mask": pad_nd([x["vid_mask"] for x in video_dicts]),
            "vid_idx": pad_nd([x["vid_idx"] for x in video_dicts]),
            "labels": pad_1d([x["labels"] for x in video_dicts]),
            "shot_ids": pad_1d([x["shot_ids"] for x in video_dicts], value=-1),
            "change_points": pad_nd([x["change_points"] for x in video_dicts], value=-1),
            "n_frame_per_seg": pad_1d([x["n_frame_per_seg"] for x in video_dicts]),
            "picks": pad_1d([x["picks"] for x in video_dicts], value=-1),
            "user_summary": pad_user_summary([x["user_summary"] for x in video_dicts]),
            "n_frames": torch.stack([x["n_frames"] for x in video_dicts]),
        }
        text_batch = {
            "dia_enc": pad_nd([x["dia_enc"] for x in text_dicts]),
            "dia_mask": pad_1d([x["dia_mask"] for x in text_dicts]),
            "word_mask": pad_nd([x["word_mask"] for x in text_dicts]),
            "labels": pad_1d([x["labels"] for x in text_dicts]),
        }
        return (
            (video_batch, text_batch),
            names,
            pad_1d(bin_indices),
            pad_1d(token_types),
            pad_1d(masks),
            pad_1d(group_idxs),
            torch.stack(subgroup_lens),
            pad_1d(labels),
        )
