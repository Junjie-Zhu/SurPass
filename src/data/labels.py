import torch

DISTANCE_BIN_START_A = 2.0
DISTANCE_BIN_END_A = 22.0
DISTANCE_BIN_WIDTH_A = 0.5
DISTANCE_BIN_NUM = 64
ATOM14_CA_INDEX = 1
ATOM14_CB_INDEX = 4


def distance_to_bins(
    distance: torch.Tensor,
    bin_start: float = DISTANCE_BIN_START_A,
    bin_end: float = DISTANCE_BIN_END_A,
    bin_count: int = DISTANCE_BIN_NUM,
) -> torch.Tensor:
    bin_limits = torch.linspace(
        float(bin_start),
        float(bin_end),
        int(bin_count) - 1,
        device=distance.device,
    )
    return torch.bucketize(distance.to(dtype=torch.float32), bin_limits).to(dtype=torch.long)


def contact_bin_count(
    contact_threshold: float,
    bin_start: float = DISTANCE_BIN_START_A,
    bin_end: float = DISTANCE_BIN_END_A,
    bin_count: int = DISTANCE_BIN_NUM,
) -> int:
    """Number of leading distance bins strictly closer than ``contact_threshold``."""
    bin_limits = torch.linspace(
        float(bin_start),
        float(bin_end),
        int(bin_count) - 1,
    )
    count = int(torch.bucketize(torch.tensor(float(contact_threshold)), bin_limits).item())
    return max(1, min(count, int(bin_count)))


def inter_chain_pair_mask(
    p1_length: int | torch.Tensor,
    p2_length: int | torch.Tensor,
    device: torch.device | None = None,
    total_length: int | None = None,
) -> torch.Tensor:
    if isinstance(p1_length, torch.Tensor) or isinstance(p2_length, torch.Tensor):
        p1 = torch.as_tensor(p1_length, dtype=torch.long)
        p2 = torch.as_tensor(p2_length, dtype=torch.long)
        if device is not None:
            p1 = p1.to(device=device)
            p2 = p2.to(device=device)
        if p1.ndim == 0:
            p1 = p1.unsqueeze(0)
        if p2.ndim == 0:
            p2 = p2.unsqueeze(0)
        if p1.shape != p2.shape:
            raise ValueError(
                f"p1_length shape {tuple(p1.shape)} does not match "
                f"p2_length shape {tuple(p2.shape)}."
            )
        batch_size = int(p1.shape[0])
        padded_total = (
            int(total_length)
            if total_length is not None
            else int((p1 + p2).max().item())
        )
        idx = torch.arange(padded_total, device=p1.device)
        i_idx = idx.view(1, padded_total, 1)
        j_idx = idx.view(1, 1, padded_total)
        length_1 = p1.view(batch_size, 1, 1)
        length_2 = p2.view(batch_size, 1, 1)
        in_p1_i = i_idx < length_1
        in_p2_i = (i_idx >= length_1) & (i_idx < length_1 + length_2)
        in_p1_j = j_idx < length_1
        in_p2_j = (j_idx >= length_1) & (j_idx < length_1 + length_2)
        mask = (in_p1_i & in_p2_j) | (in_p2_i & in_p1_j)
        if batch_size == 1 and not isinstance(p1_length, torch.Tensor):
            return mask[0]
        return mask

    p1_length = int(p1_length)
    p2_length = int(p2_length)
    total = p1_length + p2_length if total_length is None else int(total_length)
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:p1_length, p1_length : p1_length + p2_length] = True
    mask[p1_length : p1_length + p2_length, :p1_length] = True
    return mask


def build_negative_pair_labels(
    p1_positions: torch.Tensor,
    p1_mask: torch.Tensor,
    p2_positions: torch.Tensor,
    p2_mask: torch.Tensor,
    *,
    bin_start: float,
    bin_end: float,
    bin_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    p1_mask = p1_mask.to(dtype=torch.bool)
    p2_mask = p2_mask.to(dtype=torch.bool)
    p1_length = int(p1_mask.shape[0])
    p2_length = int(p2_mask.shape[0])
    p1_positions = p1_positions.to(dtype=torch.float32)
    p2_positions = p2_positions.to(dtype=torch.float32)

    combined = p1_positions.new_full(
        (p1_length + p2_length, p1_length + p2_length),
        float(bin_end) + 1.0,
    )
    combined[:p1_length, :p1_length] = torch.cdist(p1_positions, p1_positions)
    combined[p1_length:, p1_length:] = torch.cdist(p2_positions, p2_positions)

    label_bins = distance_to_bins(
        combined,
        bin_start=bin_start,
        bin_end=bin_end,
        bin_count=bin_count,
    )
    residue_mask = torch.cat([p1_mask, p2_mask], dim=0)
    label_mask = residue_mask[:, None] & residue_mask[None, :]
    return label_bins, label_mask


def inter_chain_contact_mask(
    pairwise_dist: torch.Tensor,
    pairwise_mask: torch.Tensor,
    p1_length: int,
    contact_threshold: float,
) -> torch.Tensor:
    """Inter-chain pairs closer than the contact cutoff. Shape [n1, n2]."""
    p1_length = int(p1_length)
    cross_dist = pairwise_dist[:p1_length, p1_length:]
    cross_mask = pairwise_mask[:p1_length, p1_length:]
    return (cross_dist < float(contact_threshold)) & cross_mask.to(dtype=torch.bool)


def has_inter_chain_contact(
    pairwise_dist: torch.Tensor,
    pairwise_mask: torch.Tensor,
    p1_length: int,
    contact_threshold: float,
) -> bool:
    contact = inter_chain_contact_mask(
        pairwise_dist,
        pairwise_mask,
        p1_length,
        contact_threshold,
    )
    return bool(contact.any().item())
