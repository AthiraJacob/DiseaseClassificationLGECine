"""Fixed parameters of the AHA segmentation and transmurality analysis.

Every threshold used anywhere in this package is defined here, once. There is
no configuration file and nothing is tunable at run time: these are the values
used to produce the results reported in the paper.
"""

# --- AHA 16-segment model --------------------------------------------------
# Basal and mid levels are divided into 6 sectors of 60 degrees; the apical
# level into 4 sectors of 90 degrees. Sectors are anchored at the anterior RV
# insertion point and proceed counter-clockwise.

BASAL_SEGMENTS = [1, 2, 3, 4, 5, 6]
MID_SEGMENTS = [7, 8, 9, 10, 11, 12]
APICAL_SEGMENTS = [13, 14, 15, 16]

BASAL_SEGMENT_SIZE = 60.0   # degrees; also used for the mid level
APICAL_SEGMENT_SIZE = 90.0  # degrees

SEGMENT_NAMES = {
    1: 'basal anterior',       9: 'mid inferoseptal',
    2: 'basal anteroseptal',  10: 'mid inferior',
    3: 'basal inferoseptal',  11: 'mid inferolateral',
    4: 'basal inferior',      12: 'mid anterolateral',
    5: 'basal inferolateral', 13: 'apical anterior',
    6: 'basal anterolateral', 14: 'apical septal',
    7: 'mid anterior',        15: 'apical inferior',
    8: 'mid anteroseptal',    16: 'apical lateral',
}

# Which AHA segments make up each named wall; used for the four wall_* features.
WALL_SEGMENTS = {
    'anterior': [1, 7, 13],
    'septal':   [2, 3, 8, 9, 14],
    'inferior': [4, 10, 15],
    'lateral':  [5, 6, 11, 12, 16],
}

# --- Transmurality analysis ------------------------------------------------
# Each segment is probed with N_RAYS_PER_SEGMENT radial rays cast from the LV
# centre. Along a ray the myocardium is split into three equal layers
# (endocardial, mid-myocardial, epicardial) and the scar fraction of each is
# measured.

N_RAYS_PER_SEGMENT = 15
N_RAY_SAMPLES = 200          # points sampled along each ray

LAYER_SCAR_THRESHOLD = 0.05       # scar fraction for a layer to count involved
MIN_SCAR_EXTENT_THRESHOLD = 1.0   # percent; below this a ray carries no scar

# A ray is classified transmural if ANY of three conditions holds:
#   1. scar spans at least TRANSMURAL_PCT_HIGH of the wall thickness; or
#   2. it spans at least TRANSMURAL_PCT_LOW and both the endocardial and
#      epicardial layers are involved; or
#   3. all three layers are involved, whatever the percentage.
TRANSMURAL_PCT_HIGH = 75.0
TRANSMURAL_PCT_LOW = 50.0

# When several extent classes are equally common, the dominant class is chosen
# in this order.
EXTENT_PRIORITY = ['transmural', 'endocardial', 'epicardial', 'mid-myocardial']

# --- RV insertion point ----------------------------------------------------
RVIP_SEARCH_RADIUS_PX = 3.0  # scar within this radius counts as involvement
RVIP_MAX_SNAP_PX = 15.0      # max distance to snap a point onto the boundary

# --- Slice levels ----------------------------------------------------------
LEVELS = ('basal', 'mid', 'apical')
