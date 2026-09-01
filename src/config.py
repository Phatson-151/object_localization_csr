from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_30"
)

MANIFEST_PATH = DATASET_ROOT / "manifest.csv"


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

STEP01_OUTPUT = (
    OUTPUT_ROOT
    / "step01_dataset_validation"
)

STEP02_OUTPUT = (
    OUTPUT_ROOT
    / "step02_preprocessing"
)

STEP03_OUTPUT = (
    OUTPUT_ROOT
    / "step03_supervision"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

CLASS_NAMES = {
    0: "background",
    1: "person",
    2: "car",
    3: "motorcycle",
}

TARGET_VALUES = {0, 1, 2, 3}


# ============================================================
# CITYSCAPES LABEL IDS
# ============================================================

CITYSCAPES_LABEL_IDS = {
    1: (24, 25),   # person + rider
    2: (26,),      # car
    3: (32,),      # motorcycle
}


# ============================================================
# EXPECTED DATASET SIZE
# ============================================================

EXPECTED_TOTAL_SAMPLES = 30
EXPECTED_TRAIN_SAMPLES = 24
EXPECTED_VAL_SAMPLES = 6


# ============================================================
# EXPECTED IMAGE SIZE
# ============================================================

EXPECTED_HEIGHT = 1024
EXPECTED_WIDTH = 2048


# ============================================================
# STEP 02: PREPROCESSING
# ============================================================

TIKHONOV_LAMBDA = 10.0
TIKHONOV_PADDING = 16


# ============================================================
# STEP 03: SUPERVISION PREPARATION
# ============================================================
#
# These are project-specific design choices.
# They are NOT fixed parameters from the SCSC paper.
#

SUPERVISION_RANDOM_SEED = 42

# Maximum number of positive points sampled
# from each object instance.
SUPERVISION_POINTS_PER_INSTANCE = 100

# 70% interior, 30% inner boundary.
SUPERVISION_BOUNDARY_FRACTION = 0.30

# Balanced positive / negative samples:
#
# N_negative = N_positive * ratio
#
SUPERVISION_NEGATIVE_RATIO = 1.0

# Among negative samples:
#
# 50% background
# 50% other target classes
#
SUPERVISION_OTHER_CLASS_NEGATIVE_FRACTION = 0.50

# ============================================================
# STEP 04: UNSUPERVISED CSC BASELINE
# ============================================================

STEP04_OUTPUT = (
    OUTPUT_ROOT
    / "step04_csc_baseline"
)


# ------------------------------------------------------------
# Full training configuration
# ------------------------------------------------------------

CSC_NUM_TRAIN_SAMPLES = 24

CSC_FILTER_SIZE = 11

CSC_NUM_FILTERS = 16

CSC_TRAIN_SCALE = 0.25

CSC_OUTER_ITER = 100

CSC_RANDOM_SEED = 12345


# ------------------------------------------------------------
# Sparsity parameter
# ------------------------------------------------------------
#
# IMPORTANT:
#
# The SCSC paper reports beta = 0.5, but our input signal
# has a different numerical scale because we use:
#
# RGB -> [0,1] -> grayscale -> high-pass -> scale 0.25
#
# Therefore, directly using lambda = 0.5 produced an
# all-zero sparse-code solution in our experiment.
#
# We now estimate:
#
# lambda_max = ||D^T S||_inf
#
# and use:
#
# lambda = CSC_LAMBDA_RATIO * lambda_max
#

CSC_PAPER_BETA_REFERENCE = 0.5

CSC_LAMBDA_RATIO = 0.10

# If a manually supplied lambda is >= this fraction
# of lambda_max, abort before expensive training.
CSC_LAMBDA_MAX_SAFETY = 0.95


# ------------------------------------------------------------
# ADMM parameters
# ------------------------------------------------------------

CSC_CMOD_RHO = 10.0


# ------------------------------------------------------------
# STEP 04 validation
# ------------------------------------------------------------

# L1 must become non-zero.
CSC_DEGENERATE_L1_TOL = 1e-10

# Objective must change by at least this relative amount.
CSC_MIN_OBJECTIVE_REL_CHANGE = 1e-6

# Learned dictionary should not be numerically identical
# to the initial dictionary.
CSC_MIN_DICTIONARY_REL_CHANGE = 1e-6


# ============================================================
# STEP 05: CSC SPARSE FEATURES + CLASSIFIERS
# ============================================================

STEP05_OUTPUT = (
    OUTPUT_ROOT
    / "step05_csc_features_classifiers"
)


# ------------------------------------------------------------
# CSC baseline from STEP 04
# ------------------------------------------------------------

CSC_BASELINE_RUN_NAME = (
    "csc_full_reference_k16_auto_lambda"
)


# ------------------------------------------------------------
# Sparse inference
# ------------------------------------------------------------
#
# These parameters control sparse inference with the
# LEARNED dictionary from STEP 04.
#
# MaxMainIter is a maximum budget. ConvBPDN may stop earlier
# when RelStopTol is satisfied.
#

CSC_INFERENCE_MAX_ITER = 200

CSC_INFERENCE_REL_STOP_TOL = 5e-3

# Used only when calculating coefficient density.
CSC_SPARSE_NONZERO_TOL = 1e-8


# ------------------------------------------------------------
# One-vs-All Logistic Regression
# ------------------------------------------------------------
#
# C = inverse L2 regularization strength.
#
# We keep C = 1.0 as a fixed baseline instead of tuning on the
# validation set, since validation must remain reserved for the
# final localization evaluation.
#

CSC_CLASSIFIER_C = 1.0

CSC_CLASSIFIER_MAX_ITER = 2000

# Grouped by image, so points from the same image cannot appear
# in both train and validation folds.
CSC_CLASSIFIER_CV_FOLDS = 5

# ============================================================
# STEP 06: SUPERVISED CONVOLUTIONAL SPARSE CODING
# ============================================================

STEP06_OUTPUT = (
    OUTPUT_ROOT
    / "step06_scsc_training"
)

# Use STEP 04 only as the source of:
#   - the SAME random initial dictionary D0
#   - CSC numerical scale / lambda settings
SCSC_SOURCE_STEP04_RUN = (
    "csc_full_reference_k16_auto_lambda"
)

SCSC_NUM_TRAIN_SAMPLES = 24

# Coordinate-descent outer iterations.
SCSC_OUTER_ITER = 10


# ------------------------------------------------------------
# Supervised Z update — Eq. 6
# ------------------------------------------------------------

# Previous run used only 8 ADMM iterations and 6 CG steps.
# Increase the numerical solve budget so that the supervised
# coding subproblem is solved much more accurately.
SCSC_Z_ADMM_ITER = 30
SCSC_Z_ADMM_RHO = 1.0
SCSC_Z_ADMM_TOL = 1e-4

SCSC_Z_CG_MAX_ITER = 20
SCSC_Z_CG_TOL = 1e-4

SCSC_LOGISTIC_NEWTON_ITER = 12
SCSC_LOGISTIC_NEWTON_TOL = 1e-8


# ------------------------------------------------------------
# Initial ordinary CSC coding
# ------------------------------------------------------------

SCSC_INIT_Z_MAX_ITER = 100
SCSC_INIT_Z_REL_STOP_TOL = 5e-3


# ------------------------------------------------------------
# Dictionary update — Eq. 2
# ------------------------------------------------------------

SCSC_D_ITER = 10
SCSC_D_RHO = 10.0


# ------------------------------------------------------------
# Classification term — Eq. 5 / Eq. 7
# ------------------------------------------------------------

SCSC_ALPHA = 1.0

# Project-specific numerical scaling for gamma.
#
# Initial gamma is selected so that:
#
#   gamma * supervised_term
#       ~= SCSC_GAMMA_RATIO * CSC_objective
#
# on the initial training state.
SCSC_GAMMA_RATIO = 0.10

SCSC_CLASSIFIER_MAX_ITER = 100
SCSC_CLASSIFIER_GTOL = 1e-6


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SCSC_RANDOM_SEED = 12345


# ------------------------------------------------------------
# STEP 06 safeguards / validation
# ------------------------------------------------------------

SCSC_MIN_DICTIONARY_REL_CHANGE = 1e-5
SCSC_DEGENERATE_L1_TOL = 1e-10

# Allow only tiny floating-point increases when deciding
# whether a Z / D / theta stage should be accepted.
SCSC_STAGE_OBJECTIVE_REL_TOL = 1e-5

# Final SCSC objective must not be meaningfully larger than
# the initial SCSC objective.
SCSC_FINAL_OBJECTIVE_REL_TOL = 1e-3
