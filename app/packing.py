"""
packing.py — Mango Container Packing Estimation

Estimates how many mangoes (modelled as ellipsoids) fit in a given container,
accounting for packing gaps. Also produces a 3D layout (list of mango centre
positions + semi-axes) for browser-side Three.js rendering.
"""

import math
from typing import TypedDict, Optional

# ---------------------------------------------------------------------------
# Container presets
# ---------------------------------------------------------------------------

CONTAINER_PRESETS = {
    "box_standard": {
        "label": "Standard Export Box",
        "shape": "box",
        "l_cm": 50.0,
        "w_cm": 30.0,
        "h_cm": 20.0,
        "description": "10 kg standard shipping box",
    },
    "box_small": {
        "label": "Small Carton",
        "shape": "box",
        "l_cm": 40.0,
        "w_cm": 30.0,
        "h_cm": 15.0,
        "description": "Retail 5 kg carton",
    },
    "crate_wooden": {
        "label": "Wooden Crate",
        "shape": "box",
        "l_cm": 60.0,
        "w_cm": 40.0,
        "h_cm": 30.0,
        "description": "Farm/market wooden crate",
    },
    "basket_small": {
        "label": "Small Rattan Basket",
        "shape": "cylinder",
        "l_cm": 40.0,
        "w_cm": 40.0,
        "h_cm": 25.0,
        "description": "Round basket, diameter ≈ 40 cm",
    },
    "basket_large": {
        "label": "Large Round Basket",
        "shape": "cylinder",
        "l_cm": 60.0,
        "w_cm": 60.0,
        "h_cm": 35.0,
        "description": "Market/harvest basket, diameter ≈ 60 cm",
    },
    "custom": {
        "label": "Custom Container",
        "shape": "box",
        "l_cm": None,
        "w_cm": None,
        "h_cm": None,
        "description": "User-defined box dimensions",
    },
}

# ---------------------------------------------------------------------------
# Helper geometry
# ---------------------------------------------------------------------------

def ellipsoid_volume(a: float, b: float, c: float) -> float:
    """Volume of an ellipsoid with semi-axes a, b, c (any consistent units)."""
    return (4.0 / 3.0) * math.pi * a * b * c


def container_volume(shape: str, l: float, w: float, h: float) -> float:
    """Usable internal volume of a container (cm³)."""
    if shape == "cylinder":
        r = min(l, w) / 2.0
        return math.pi * r * r * h
    return l * w * h  # box


# ---------------------------------------------------------------------------
# Quick estimate
# ---------------------------------------------------------------------------

# Random packing efficiency for ellipsoid-like objects (~0.64 for sphere,
# slightly lower for elongated shapes — we use 0.60 as an empirical constant).
RANDOM_PACKING_EFF = 0.60


def estimate_count(
    mango_h: float,
    mango_w: float,
    mango_t: float,
    container_id: str,
    custom_l: Optional[float] = None,
    custom_w: Optional[float] = None,
    custom_h: Optional[float] = None,
) -> dict:
    """
    High-level estimation of how many mangoes fit in a container.

    Mango dimensions are in **cm**.  Returns a result dict.
    """
    preset = CONTAINER_PRESETS.get(container_id)
    if preset is None:
        raise ValueError(f"Unknown container_id: {container_id!r}")

    shape = preset["shape"]

    if container_id == "custom":
        if any(v is None for v in (custom_l, custom_w, custom_h)):
            raise ValueError("custom_l, custom_w, custom_h are required for custom containers.")
        l_cm, w_cm, h_cm = float(custom_l), float(custom_w), float(custom_h)
        shape = "box"
    else:
        l_cm = preset["l_cm"]
        w_cm = preset["w_cm"]
        h_cm = preset["h_cm"]

    # Mango ellipsoid semi-axes
    a, b, c = mango_h / 2, mango_w / 2, mango_t / 2

    mango_vol = ellipsoid_volume(a, b, c)
    cont_vol = container_volume(shape, l_cm, w_cm, h_cm)

    if mango_vol <= 0:
        raise ValueError("Mango volume is zero or negative.")

    count_estimate = int(RANDOM_PACKING_EFF * cont_vol / mango_vol)

    # Layer-by-layer breakdown using bounding box footprint
    layout = simulate_layout(mango_h, mango_w, mango_t, shape, l_cm, w_cm, h_cm)

    return {
        "count_estimate": max(count_estimate, layout["count_placed"]),
        "count_placed": layout["count_placed"],
        "packing_efficiency_pct": round(RANDOM_PACKING_EFF * 100, 1),
        "layers": layout["layers"],
        "per_layer": layout["per_layer"],
        "container_volume_cm3": round(cont_vol, 2),
        "mango_volume_cm3": round(mango_vol, 2),
        "container": {
            "id": container_id,
            "label": preset["label"] if container_id != "custom" else "Custom Container",
            "shape": shape,
            "l_cm": l_cm,
            "w_cm": w_cm,
            "h_cm": h_cm,
        },
        "mango": {
            "height_cm": mango_h,
            "width_cm": mango_w,
            "thickness_cm": mango_t,
        },
        "positions": layout["positions"],  # list of {x, y, z, rx, ry, rz} in cm
    }


# ---------------------------------------------------------------------------
# Layer-by-layer layout simulation
# ---------------------------------------------------------------------------

def simulate_layout(
    mango_h: float,
    mango_w: float,
    mango_t: float,
    shape: str,
    l_cm: float,
    w_cm: float,
    h_cm: float,
    max_mangoes: int = 2000,
) -> dict:
    """
    Simulate a grid-based layer packing with HCP (hexagonal close-packing)
    offset on alternating layers.

    Mangoes are laid on their flattest face: oriented so that height (longest)
    goes along the container length, width goes across, and thickness is the
    stacking direction.

    Returns a dict with positions list (for Three.js) and layer counts.
    """
    # Ellipsoid overlap and nesting factors representing natural close packing
    SPACING_FACTOR = 0.82
    VERT_NEST_FACTOR = 0.80

    foot_x = mango_h * SPACING_FACTOR   # along container length
    foot_y = mango_w * SPACING_FACTOR   # along container width
    stack_z = mango_t * VERT_NEST_FACTOR  # stacking height

    # For cylinders, use the full dimension since we filter coordinates explicitly
    eff_l = l_cm
    eff_w = w_cm

    positions = []
    layer_counts = []
    layer = 0
    z = stack_z / 2.0  # centre z of first layer

    while z + stack_z / 2.0 <= h_cm and len(positions) < max_mangoes:
        # HCP offset on odd layers
        x_offset = (foot_x / 2.0) if (layer % 2 == 1) else 0.0
        y_offset = (foot_y / 2.0) if (layer % 2 == 1) else 0.0

        count_this_layer = 0
        x = foot_x / 2.0 + x_offset
        while x + foot_x / 2.0 <= eff_l and len(positions) < max_mangoes:
            y = foot_y / 2.0 + y_offset
            while y + foot_y / 2.0 <= eff_w and len(positions) < max_mangoes:
                # For cylinder: check if mango centre is inside the circle
                if shape == "cylinder":
                    r = min(l_cm, w_cm) / 2.0
                    cx, cy = l_cm / 2.0, w_cm / 2.0
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    # Allow if the mango's bounding circle fits inside the container circle
                    mango_r = math.sqrt((foot_x / 2) ** 2 + (foot_y / 2) ** 2)
                    if dist + mango_r > r:
                        y += foot_y
                        continue

                positions.append({
                    "x": round(x - eff_l / 2, 3),   # centred on origin
                    "y": round(y - eff_w / 2, 3),
                    "z": round(z - h_cm / 2, 3),
                    # Semi-axes for Three.js SphereGeometry + scale
                    "rx": round(mango_h / 2, 3),
                    "ry": round(mango_w / 2, 3),
                    "rz": round(mango_t / 2, 3),
                })
                count_this_layer += 1
                y += foot_y
            x += foot_x

        layer_counts.append(count_this_layer)
        layer += 1
        z += stack_z

    return {
        "count_placed": len(positions),
        "layers": len(layer_counts),
        "per_layer": layer_counts,
        "positions": positions,
    }
