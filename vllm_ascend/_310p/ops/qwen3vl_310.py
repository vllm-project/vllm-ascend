import torch


# 310P: non_blocking H2D copy in rot_pos_emb can race with subsequent indexing.
def rot_pos_emb_310(self, grid_thw: list[list[int]]):
    max_grid_size = max(max(h, w) for _, h, w in grid_thw)
    pos_ids = [
        self.rot_pos_ids(h, w, self.spatial_merge_size)
        if t == 1
        else self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
        for t, h, w in grid_thw
    ]
    pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=False)

    cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)
    cos_combined = cos[pos_ids].flatten(1)
    sin_combined = sin[pos_ids].flatten(1)
    return cos_combined, sin_combined
