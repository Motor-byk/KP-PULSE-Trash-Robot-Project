"""Object class -> disposal bin mapping.

Perception predicts *object types*; this table turns them into bin decisions.
Bin rules are regional policy, so they live here rather than in the model
weights -- changing how your area sorts trash means editing this file, not
retraining.
"""

# Bin identifiers
RECYCLE = "recycle"
COMPOST = "compost"
WASTE = "waste"
UNKNOWN = "unknown"

# BGR draw colors, for cv2
BIN_COLORS = {
    RECYCLE: (255, 128, 0),    # blue
    COMPOST: (0, 200, 0),      # green
    WASTE: (60, 60, 60),       # dark grey
    UNKNOWN: (0, 0, 255),      # red
}


# --------------------------------------------------------------------------
# Custom model classes (Step 2 of the plan).
# Keep these keys identical to the `names:` list in the dataset data.yaml.
# --------------------------------------------------------------------------

CUSTOM_CLASS_TO_BIN = {
    "plastic_bottle": RECYCLE,
    "metal_can": RECYCLE,
    "glass_bottle": RECYCLE,
    "cardboard": RECYCLE,
    "paper_clean": RECYCLE,
    "beverage_carton": RECYCLE,

    "food_scrap": COMPOST,
    "banana_peel": COMPOST,
    "apple_core": COMPOST,
    "soiled_paper": COMPOST,

    "plastic_wrapper": WASTE,
    "styrofoam": WASTE,
    "plastic_utensil": WASTE,
    "ceramic": WASTE,
}


# --------------------------------------------------------------------------
# COCO classes, for the zero-shot prototype using stock yolo11s.pt.
# Lets the full pipeline run before any data is captured or labeled.
# --------------------------------------------------------------------------

COCO_CLASS_TO_BIN = {
    "bottle": RECYCLE,
    "wine glass": RECYCLE,
    "book": RECYCLE,       # stand-in for paper/cardboard, which COCO lacks

    "banana": COMPOST,
    "apple": COMPOST,
    "orange": COMPOST,
    "broccoli": COMPOST,
    "carrot": COMPOST,
    "pizza": COMPOST,
    "donut": COMPOST,
    "cake": COMPOST,
    "sandwich": COMPOST,
    "hot dog": COMPOST,

    "cup": WASTE,          # assumed disposable/lined rather than a ceramic mug
    "bowl": WASTE,         # assumed ceramic
    "toothbrush": WASTE,
}

# COCO indices for the classes above, passed as `classes=` so the detector
# never even reports the categories we have no rule for.
COCO_MAPPED_IDS = [39, 40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 73, 79]

# Detected on purpose but deliberately absent from COCO_CLASS_TO_BIN, so
# `bin_for()` returns UNKNOWN for them. These exercise the safety path: a
# stably tracked object that must still never become a grasp target.
# Do NOT add bin rules for these -- that would delete the only live test of it.
COCO_DISTRACTOR_IDS = [67, 76]   # cell phone, scissors

COCO_CLASS_IDS = COCO_MAPPED_IDS + COCO_DISTRACTOR_IDS


def bin_for(class_name, prototype=False):
    """Map a predicted class name to a bin, defaulting to UNKNOWN.

    Anything unmapped deliberately falls through to UNKNOWN rather than
    guessing -- the arm should never act on a class this table does not know.
    """
    table = COCO_CLASS_TO_BIN if prototype else CUSTOM_CLASS_TO_BIN
    return table.get(class_name, UNKNOWN)
