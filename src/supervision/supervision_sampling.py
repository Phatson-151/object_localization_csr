import numpy as np

from scipy.ndimage import (
    distance_transform_edt,
)

from src.config import (
    CITYSCAPES_LABEL_IDS,
)


# ============================================================
# POINT TYPE CODES
# ============================================================
#
# Positive:
#
#   1 = interior positive
#   2 = boundary positive
#
# Negative:
#
#  -1 = background negative
#  -2 = other-class negative
#

POINT_INTERIOR_POSITIVE = 1
POINT_BOUNDARY_POSITIVE = 2

POINT_BACKGROUND_NEGATIVE = -1
POINT_OTHER_CLASS_NEGATIVE = -2


def get_instance_ids(
    instance_mask: np.ndarray,
    target_class_id: int,
) -> list:
    """
    Return Cityscapes instance IDs belonging
    to one target class.

    Example
    -------
    target class 1:
        Cityscapes person (24)
        Cityscapes rider  (25)

    target class 2:
        car (26)

    target class 3:
        motorcycle (32)
    """

    label_ids = (
        CITYSCAPES_LABEL_IDS[
            target_class_id
        ]
    )

    valid = (
        instance_mask >= 1000
    )

    unique_ids = np.unique(
        instance_mask[valid]
    )

    result = []

    for instance_id in unique_ids:

        cityscapes_class = int(
            instance_id // 1000
        )

        if cityscapes_class in label_ids:

            result.append(
                int(instance_id)
            )

    return result


def _randomized_distance_order(
    distances: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sort pixels by distance from the object
    boundary.

    Small distance:
        near boundary

    Large distance:
        deeper inside object

    Pixel order is randomized first so that
    equal-distance pixels do not always use
    deterministic raster order.
    """

    n = len(distances)

    permutation = rng.permutation(
        n
    )

    sorted_local = np.argsort(
        distances[permutation],
        kind="stable",
    )

    return permutation[
        sorted_local
    ]


def sample_instance_positive_points(
    instance_binary_mask: np.ndarray,
    rng: np.random.Generator,
    max_points: int,
    boundary_fraction: float,
) -> dict:
    """
    Sample positive supervision points from
    one object instance.

    Strategy
    --------
    Up to max_points are selected.

    Approximately:
        70% interior
        30% inner boundary

    Boundary is defined using distance to
    the object's boundary.

    No points outside the object are used.
    No duplicated coordinates are created.
    """

    if instance_binary_mask.ndim != 2:

        raise ValueError(
            "Instance binary mask must "
            "be a 2D array."
        )

    coordinates = np.argwhere(
        instance_binary_mask
    ).astype(
        np.int32
    )

    object_pixels = len(
        coordinates
    )

    if object_pixels == 0:

        return {
            "coordinates":
                np.empty(
                    (0, 2),
                    dtype=np.int32,
                ),

            "types":
                np.empty(
                    (0,),
                    dtype=np.int8,
                ),

            "object_pixels":
                0,

            "selected_total":
                0,

            "interior_selected":
                0,

            "boundary_selected":
                0,

            "interior_region_pixels":
                0,

            "boundary_region_pixels":
                0,
        }

    # --------------------------------------------------------
    # NUMBER OF SUPERVISION POINTS
    # --------------------------------------------------------

    n_total = min(
        int(max_points),
        object_pixels,
    )

    if n_total == 1:

        n_boundary = 1
        n_interior = 0

    else:

        n_boundary = int(
            round(
                n_total
                * boundary_fraction
            )
        )

        n_boundary = max(
            1,
            min(
                n_boundary,
                n_total - 1,
            ),
        )

        n_interior = (
            n_total
            - n_boundary
        )

    # --------------------------------------------------------
    # DISTANCE FROM OBJECT BOUNDARY
    # --------------------------------------------------------

    distance_map = (
        distance_transform_edt(
            instance_binary_mask
        )
    )

    distances = distance_map[
        instance_binary_mask
    ]

    ranked_indices = (
        _randomized_distance_order(
            distances,
            rng,
        )
    )

    # --------------------------------------------------------
    # DEFINE INNER-BOUNDARY REGION
    #
    # The 30% of object pixels closest to
    # the edge form the boundary candidate
    # region.
    #
    # Remaining pixels form the interior.
    # --------------------------------------------------------

    boundary_region_size = int(
        np.ceil(
            object_pixels
            * boundary_fraction
        )
    )

    boundary_region_size = max(
        n_boundary,
        boundary_region_size,
    )

    # We must leave enough pixels for the
    # desired number of interior samples.

    maximum_boundary_region = (
        object_pixels
        - n_interior
    )

    boundary_region_size = min(
        boundary_region_size,
        maximum_boundary_region,
    )

    boundary_indices = (
        ranked_indices[
            :boundary_region_size
        ]
    )

    interior_indices = (
        ranked_indices[
            boundary_region_size:
        ]
    )

    # --------------------------------------------------------
    # SAMPLE WITHOUT REPLACEMENT
    # --------------------------------------------------------

    if n_boundary > 0:

        selected_boundary = rng.choice(
            boundary_indices,
            size=n_boundary,
            replace=False,
        )

        boundary_coordinates = (
            coordinates[
                selected_boundary
            ]
        )

    else:

        boundary_coordinates = (
            np.empty(
                (0, 2),
                dtype=np.int32,
            )
        )

    if n_interior > 0:

        selected_interior = rng.choice(
            interior_indices,
            size=n_interior,
            replace=False,
        )

        interior_coordinates = (
            coordinates[
                selected_interior
            ]
        )

    else:

        interior_coordinates = (
            np.empty(
                (0, 2),
                dtype=np.int32,
            )
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    selected_coordinates = np.vstack(
        [
            interior_coordinates,
            boundary_coordinates,
        ]
    ).astype(
        np.int32
    )

    selected_types = np.concatenate(
        [
            np.full(
                len(interior_coordinates),
                POINT_INTERIOR_POSITIVE,
                dtype=np.int8,
            ),

            np.full(
                len(boundary_coordinates),
                POINT_BOUNDARY_POSITIVE,
                dtype=np.int8,
            ),
        ]
    )

    return {
        "coordinates":
            selected_coordinates,

        "types":
            selected_types,

        "object_pixels":
            int(object_pixels),

        "selected_total":
            int(n_total),

        "interior_selected":
            int(
                len(interior_coordinates)
            ),

        "boundary_selected":
            int(
                len(boundary_coordinates)
            ),

        "interior_region_pixels":
            int(
                len(interior_indices)
            ),

        "boundary_region_pixels":
            int(
                len(boundary_indices)
            ),
    }


def _sample_mask_coordinates(
    mask: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly sample pixel coordinates from
    a binary mask without replacement.
    """

    if n_samples <= 0:

        return np.empty(
            (0, 2),
            dtype=np.int32,
        )

    flat_candidates = np.flatnonzero(
        mask
    )

    if len(flat_candidates) < n_samples:

        raise ValueError(
            f"Requested {n_samples} pixels "
            f"but only "
            f"{len(flat_candidates)} "
            f"candidate pixels exist."
        )

    selected_flat = rng.choice(
        flat_candidates,
        size=n_samples,
        replace=False,
    )

    rows, cols = np.unravel_index(
        selected_flat,
        mask.shape,
    )

    return np.column_stack(
        [
            rows,
            cols,
        ]
    ).astype(
        np.int32
    )


def sample_balanced_negative_points(
    target_mask: np.ndarray,
    target_class_id: int,
    n_negative: int,
    other_class_fraction: float,
    rng: np.random.Generator,
) -> dict:
    """
    Sample negative supervision points for
    one One-vs-All classifier.

    Example: Person vs All

    Positive:
        person pixels

    Negative:
        background
        car
        motorcycle

    The requested negative set is divided
    between background and other target
    classes when enough pixels are available.
    """

    if n_negative <= 0:

        return {
            "coordinates":
                np.empty(
                    (0, 2),
                    dtype=np.int32,
                ),

            "types":
                np.empty(
                    (0,),
                    dtype=np.int8,
                ),

            "background_selected":
                0,

            "other_class_selected":
                0,
        }

    background_mask = (
        target_mask == 0
    )

    other_class_mask = (
        (target_mask != 0)
        &
        (
            target_mask
            != target_class_id
        )
    )

    n_background_available = int(
        np.count_nonzero(
            background_mask
        )
    )

    n_other_available = int(
        np.count_nonzero(
            other_class_mask
        )
    )

    total_available = (
        n_background_available
        +
        n_other_available
    )

    if total_available < n_negative:

        raise ValueError(
            "Not enough negative pixels "
            f"for class {target_class_id}: "
            f"requested={n_negative}, "
            f"available={total_available}"
        )

    desired_other = int(
        round(
            n_negative
            * other_class_fraction
        )
    )

    desired_background = (
        n_negative
        - desired_other
    )

    # --------------------------------------------------------
    # FIRST ALLOCATION
    # --------------------------------------------------------

    n_other = min(
        desired_other,
        n_other_available,
    )

    n_background = min(
        desired_background,
        n_background_available,
    )

    remaining = (
        n_negative
        - n_other
        - n_background
    )

    # --------------------------------------------------------
    # FILL SHORTAGE FROM AVAILABLE POOL
    # --------------------------------------------------------

    if remaining > 0:

        additional_other = min(
            remaining,
            n_other_available
            - n_other,
        )

        n_other += (
            additional_other
        )

        remaining -= (
            additional_other
        )

    if remaining > 0:

        additional_background = min(
            remaining,
            n_background_available
            - n_background,
        )

        n_background += (
            additional_background
        )

        remaining -= (
            additional_background
        )

    if remaining != 0:

        raise RuntimeError(
            "Unable to construct the "
            "requested negative sample set."
        )

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    background_coordinates = (
        _sample_mask_coordinates(
            background_mask,
            n_background,
            rng,
        )
    )

    other_coordinates = (
        _sample_mask_coordinates(
            other_class_mask,
            n_other,
            rng,
        )
    )

    coordinates = np.vstack(
        [
            background_coordinates,
            other_coordinates,
        ]
    ).astype(
        np.int32
    )

    types = np.concatenate(
        [
            np.full(
                len(
                    background_coordinates
                ),
                POINT_BACKGROUND_NEGATIVE,
                dtype=np.int8,
            ),

            np.full(
                len(
                    other_coordinates
                ),
                POINT_OTHER_CLASS_NEGATIVE,
                dtype=np.int8,
            ),
        ]
    )

    return {
        "coordinates":
            coordinates,

        "types":
            types,

        "background_selected":
            int(
                len(
                    background_coordinates
                )
            ),

        "other_class_selected":
            int(
                len(
                    other_coordinates
                )
            ),
    }


def validate_class_supervision(
    coordinates: np.ndarray,
    labels: np.ndarray,
    point_types: np.ndarray,
    target_mask: np.ndarray,
    target_class_id: int,
) -> list:
    """
    Validate one class-specific supervision
    set.
    """

    errors = []

    n = len(
        coordinates
    )

    if labels.shape != (n,):

        errors.append(
            "labels length does not match "
            "coordinates"
        )

    if point_types.shape != (n,):

        errors.append(
            "point_types length does not "
            "match coordinates"
        )

    unique_labels = set(
        np.unique(
            labels
        ).tolist()
    )

    if not unique_labels.issubset(
        {-1, 1}
    ):

        errors.append(
            "labels contain values other "
            "than -1 and +1"
        )

    if n == 0:

        errors.append(
            "empty supervision set"
        )

        return errors

    rows = coordinates[:, 0]
    cols = coordinates[:, 1]

    if (
        np.any(rows < 0)
        or np.any(
            rows
            >= target_mask.shape[0]
        )
        or np.any(cols < 0)
        or np.any(
            cols
            >= target_mask.shape[1]
        )
    ):

        errors.append(
            "coordinates outside image "
            "bounds"
        )

        return errors

    pixel_classes = (
        target_mask[
            rows,
            cols,
        ]
    )

    positive = (
        labels == 1
    )

    negative = (
        labels == -1
    )

    if not np.all(
        pixel_classes[
            positive
        ]
        == target_class_id
    ):

        errors.append(
            "one or more positive points "
            "are outside the target class"
        )

    if np.any(
        pixel_classes[
            negative
        ]
        == target_class_id
    ):

        errors.append(
            "one or more negative points "
            "fall inside the target class"
        )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    flat = np.ravel_multi_index(
        (
            rows,
            cols,
        ),
        target_mask.shape,
    )

    if len(
        np.unique(flat)
    ) != len(flat):

        errors.append(
            "duplicate supervision "
            "coordinates detected"
        )

    return errors
