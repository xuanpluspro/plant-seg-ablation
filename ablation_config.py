"""Pure-Python ablation definitions: each flag controls every level of a module."""
PROTOCOL = "bimseg-three-module-v1"
VARIANTS = {
    "baseline": (False, False, False),
    "msca": (True, False, False),
    "drcf": (False, True, False),
    "dcam": (False, False, True),
    "msca_drcf": (True, True, False),
    "msca_dcam": (True, False, True),
    "drcf_dcam": (False, True, True),
    "full": (True, True, True),
}
FLAG_NAMES = ("use_msca", "use_drcf", "use_dcam")


def variant_flags(name):
    return dict(zip(FLAG_NAMES, VARIANTS[name]))


def canonical_model_config(config):
    """Add legacy full-model defaults for resume comparisons, never alter weights."""
    config = dict(config)
    if not config.get("use_dynamic_fusion", True):
        raise ValueError("Legacy FEM models are not a DCAM-off ablation; use the fixed baseline replacement")
    for flag in FLAG_NAMES:
        config.setdefault(flag, True)
        if type(config[flag]) is not bool:
            raise ValueError(f"{flag} must be a boolean")
    return config


def describe_ablation(config):
    config = canonical_model_config(config)
    flags = {flag: config[flag] for flag in FLAG_NAMES}
    values = tuple(flags[flag] for flag in FLAG_NAMES)
    name = next(name for name, combination in VARIANTS.items() if combination == values)
    return dict(protocol=PROTOCOL, variant=name, **flags)
