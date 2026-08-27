import re
from collections import defaultdict
import numpy as np


def _subset_inplace(dataset, indices):
    """Keep only selected indices (sync image_list/disp_list/extra_info)."""
    dataset.image_list = [dataset.image_list[i] for i in indices]
    dataset.disparity_list = [dataset.disparity_list[i] for i in indices]
    if hasattr(dataset, "extra_info") and len(dataset.extra_info) > 0:
        dataset.extra_info = [dataset.extra_info[i] for i in indices]
    return dataset


def subset(dataset, max_samples=None, seed=0):
    """Randomly downsample to max_samples (reproducible)."""
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(dataset))[:max_samples].tolist()
    return _subset_inplace(dataset, idx)


def subset_by_group(dataset, group_fn, max_per_group=None, seed=0, max_total=None):
    """
    Per-group cap then (optional) global cap. Reproducible.
    group_fn(im1, im2, disp) -> key
    """
    if max_per_group is None or max_per_group <= 0:
        return subset(dataset, max_total, seed=seed) if max_total else dataset

    rng = np.random.RandomState(seed)
    buckets = defaultdict(list)

    for i, (im1, im2) in enumerate(dataset.image_list):
        disp = dataset.disparity_list[i]
        k = group_fn(im1, im2, disp)
        buckets[k].append(i)

    kept = []
    for _, idxs in buckets.items():
        rng.shuffle(idxs)
        kept.extend(idxs[:max_per_group])

    rng.shuffle(kept)
    if max_total is not None and 0 < max_total < len(kept):
        kept = kept[:max_total]

    return _subset_inplace(dataset, kept)


# ---------------------------
# Group key parsers
# ---------------------------

_RE_TA = re.compile(r"/TartanAir/([^/]+)/\1/(Easy|Hard)/([^/]+)/")
def tartanair_group_key(im1, im2, disp):
    # example:
    # .../TartanAir/abandonedfactory/abandonedfactory/Easy/P000/image_left/000000_left.png
    p = im1.replace("\\", "/")
    m = _RE_TA.search(p)
    if m:
        scene, mode, pseq = m.group(1), m.group(2), m.group(3)
        return f"{scene}/{mode}/{pseq}"
    return "unknown"


_RE_CRES = re.compile(r"/crestereo_dataset/(hole|reflective|shapenet|tree)/([0-9]+)/")
def crestereo_group_key(im1, im2, disp):
    p = im1.replace("\\", "/")
    m = _RE_CRES.search(p)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return "unknown"


_RE_FT_MIXED = re.compile(r"/FallingThings/mixed/([^/]+)/")
def fallingthings_mixed_group_key(im1, im2, disp):
    # example:
    # .../FallingThings/mixed/kitchen_0/000000.left.jpg
    p = im1.replace("\\", "/")
    m = _RE_FT_MIXED.search(p)
    if m:
        return m.group(1)
    return "unknown"


def filter_out_crestereo_tree(dataset):
    """In-place remove any sample whose left image path contains /tree/."""
    keep = []
    for i, (im1, _) in enumerate(dataset.image_list):
        if "/tree/" in im1.replace("\\", "/"):
            continue
        keep.append(i)
    return _subset_inplace(dataset, keep)


def build_foundation_capped_mix(
    sceneflow,
    tartanair,
    crestereo,
    fallingthings,
    vkitti2,
    instereo2k,
    *,
    seed=666,
    tartanair_max_total=35000,
    tartanair_max_per_group=100,
    crestereo_max_total=39000,
    crestereo_max_per_group=1300,
    exclude_crestereo_tree=True,
    fallingthings_max_total=9000,
    fallingthings_max_per_group=600,
):
    """
    Return (mixed_dataset, stats_dict). Datasets are modified in-place (capped).
    """

    # TartanAir: per (scene/mode/Pxxx) cap then total cap
    tartanair = subset_by_group(
        tartanair,
        tartanair_group_key,
        max_per_group=tartanair_max_per_group,
        seed=seed,
        max_total=tartanair_max_total,
    )

    # CREStereo: optionally exclude tree, then per (scene/digit) cap
    if exclude_crestereo_tree:
        crestereo = filter_out_crestereo_tree(crestereo)

    crestereo = subset_by_group(
        crestereo,
        crestereo_group_key,
        max_per_group=crestereo_max_per_group,
        seed=seed + 1,
        max_total=crestereo_max_total,
    )

    # FallingThings: your loader only loads mixed; do per folder cap
    fallingthings = subset_by_group(
        fallingthings,
        fallingthings_mixed_group_key,
        max_per_group=fallingthings_max_per_group,
        seed=seed + 2,
        max_total=fallingthings_max_total,
    )

    mixed = sceneflow + tartanair + crestereo + fallingthings + vkitti2 + instereo2k

    stats = {
        "sceneflow": len(sceneflow),
        "tartanair": len(tartanair),
        "crestereo": len(crestereo),
        "fallingthings": len(fallingthings),
        "vkitti2": len(vkitti2),
        "instereo2k": len(instereo2k),
        "total": len(mixed),
    }
    return mixed, stats