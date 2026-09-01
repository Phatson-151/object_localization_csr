import sys

import numpy as np
import pandas as pd

from src.config import (
    DATASET_ROOT,
    MANIFEST_PATH,
    STEP01_OUTPUT,
    TARGET_VALUES,
    CLASS_NAMES,
    EXPECTED_TOTAL_SAMPLES,
    EXPECTED_TRAIN_SAMPLES,
    EXPECTED_VAL_SAMPLES,
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
)

from src.data.dataset_utils import (
    get_sample_paths,
    load_sample,
    count_target_pixels,
    count_instances,
    build_expected_target_from_labels,
)

from src.utils.output_utils import (
    create_output_dir,
    step_filename,
)


STEP_NUMBER = 1


# ============================================================
# MANIFEST COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "split",
    "city",
    "sample_id",

    "person_pixels",
    "car_pixels",
    "motorcycle_pixels",

    "person_instances",
    "car_instances",
    "motorcycle_instances",
}


PIXEL_COLUMNS = {
    1: "person_pixels",
    2: "car_pixels",
    3: "motorcycle_pixels",
}


INSTANCE_COLUMNS = {
    1: "person_instances",
    2: "car_instances",
    3: "motorcycle_instances",
}


def add_error(
    errors: list,
    sample_id: str,
    error_type: str,
    message: str,
):
    """
    Add one validation error.
    """

    errors.append(
        {
            "sample_id": sample_id,
            "error_type": error_type,
            "message": message,
        }
    )


def main():

    # ========================================================
    # PREPARE OUTPUT DIRECTORY
    # ========================================================

    output_dir = create_output_dir(
        STEP01_OUTPUT
    )

    report_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "validation_report",
            "txt",
        )
    )

    dataset_summary_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "dataset_summary",
            "csv",
        )
    )

    sample_summary_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "sample_summary",
            "csv",
        )
    )

    errors_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "validation_errors",
            "csv",
        )
    )

    print("=" * 72)
    print("STEP 01: DATASET VALIDATION")
    print("=" * 72)

    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Manifest     : {MANIFEST_PATH}")
    print()

    # ========================================================
    # CHECK MANIFEST
    # ========================================================

    if not MANIFEST_PATH.exists():

        message = (
            f"Manifest file not found: "
            f"{MANIFEST_PATH}"
        )

        print("FAILED")
        print(message)

        report_path.write_text(
            message,
            encoding="utf-8",
        )

        raise SystemExit(1)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(manifest.columns)
    )

    if missing_columns:

        message = (
            "Missing required manifest columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

        print("FAILED")
        print(message)

        report_path.write_text(
            message,
            encoding="utf-8",
        )

        raise SystemExit(1)

    # ========================================================
    # GLOBAL DATASET CHECK
    # ========================================================

    errors = []

    total_samples = len(manifest)

    train_samples = int(
        (manifest["split"] == "train").sum()
    )

    val_samples = int(
        (manifest["split"] == "val").sum()
    )

    if total_samples != EXPECTED_TOTAL_SAMPLES:

        add_error(
            errors,
            "__DATASET__",
            "sample_count",
            (
                f"Expected "
                f"{EXPECTED_TOTAL_SAMPLES} samples, "
                f"found {total_samples}"
            ),
        )

    if train_samples != EXPECTED_TRAIN_SAMPLES:

        add_error(
            errors,
            "__DATASET__",
            "train_count",
            (
                f"Expected "
                f"{EXPECTED_TRAIN_SAMPLES} train samples, "
                f"found {train_samples}"
            ),
        )

    if val_samples != EXPECTED_VAL_SAMPLES:

        add_error(
            errors,
            "__DATASET__",
            "val_count",
            (
                f"Expected "
                f"{EXPECTED_VAL_SAMPLES} validation samples, "
                f"found {val_samples}"
            ),
        )

    # ========================================================
    # SAMPLE VALIDATION
    # ========================================================

    sample_rows = []

    for index, row in manifest.iterrows():

        split = str(row["split"])
        sample_id = str(
            row["sample_id"]
        )

        print(
            f"[{index + 1:02d}/{total_samples:02d}] "
            f"{split:<5} | {sample_id}"
        )

        sample_errors_before = len(errors)

        # ----------------------------------------------------
        # VALID SPLIT
        # ----------------------------------------------------

        if split not in {
            "train",
            "val",
        }:

            add_error(
                errors,
                sample_id,
                "invalid_split",
                f"Unexpected split: {split}",
            )

            continue

        # ----------------------------------------------------
        # FILE EXISTENCE
        # ----------------------------------------------------

        paths = get_sample_paths(
            split=split,
            sample_id=sample_id,
        )

        missing_files = [
            str(path)
            for path in paths.values()
            if not path.exists()
        ]

        if missing_files:

            for path in missing_files:

                add_error(
                    errors,
                    sample_id,
                    "missing_file",
                    path,
                )

            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "city": row["city"],
                    "height": None,
                    "width": None,
                    "person_pixels": None,
                    "car_pixels": None,
                    "motorcycle_pixels": None,
                    "person_instances": None,
                    "car_instances": None,
                    "motorcycle_instances": None,
                    "target_values": None,
                    "status": "FAIL",
                }
            )

            continue

        # ----------------------------------------------------
        # LOAD SAMPLE
        # ----------------------------------------------------

        try:

            sample = load_sample(
                split=split,
                sample_id=sample_id,
            )

        except Exception as exc:

            add_error(
                errors,
                sample_id,
                "load_error",
                str(exc),
            )

            continue

        image = sample["image"]
        label = sample["label"]
        instance = sample["instance"]
        target = sample["target"]

        # ----------------------------------------------------
        # IMAGE DIMENSIONS
        # ----------------------------------------------------

        height, width = (
            image.shape[:2]
        )

        if image.shape != (
            EXPECTED_HEIGHT,
            EXPECTED_WIDTH,
            3,
        ):

            add_error(
                errors,
                sample_id,
                "image_shape",
                (
                    f"Expected "
                    f"({EXPECTED_HEIGHT}, "
                    f"{EXPECTED_WIDTH}, 3), "
                    f"found {image.shape}"
                ),
            )

        expected_mask_shape = (
            height,
            width,
        )

        for mask_name, mask in [
            ("label", label),
            ("instance", instance),
            ("target", target),
        ]:

            if mask.shape != expected_mask_shape:

                add_error(
                    errors,
                    sample_id,
                    f"{mask_name}_shape",
                    (
                        f"{mask_name} shape "
                        f"{mask.shape} != "
                        f"{expected_mask_shape}"
                    ),
                )

        # ----------------------------------------------------
        # TARGET VALUES
        # ----------------------------------------------------

        unique_target_values = set(
            np.unique(target).tolist()
        )

        if not unique_target_values.issubset(
            TARGET_VALUES
        ):

            invalid_values = (
                unique_target_values
                - TARGET_VALUES
            )

            add_error(
                errors,
                sample_id,
                "target_values",
                (
                    "Unexpected target values: "
                    f"{sorted(invalid_values)}"
                ),
            )

        # ----------------------------------------------------
        # VERIFY TARGET MASK AGAINST ORIGINAL LABEL IDS
        # ----------------------------------------------------

        if (
            label.shape
            == target.shape
        ):

            expected_target = (
                build_expected_target_from_labels(
                    label
                )
            )

            mismatch_pixels = int(
                np.count_nonzero(
                    expected_target != target
                )
            )

            if mismatch_pixels > 0:

                add_error(
                    errors,
                    sample_id,
                    "target_label_mapping",
                    (
                        f"{mismatch_pixels} pixels "
                        "do not match expected "
                        "Cityscapes label mapping"
                    ),
                )

        # ----------------------------------------------------
        # PIXEL COUNTS
        # ----------------------------------------------------

        observed_pixels = {}

        for class_id in (1, 2, 3):

            observed = (
                count_target_pixels(
                    target,
                    class_id,
                )
            )

            observed_pixels[
                class_id
            ] = observed

            column = PIXEL_COLUMNS[
                class_id
            ]

            expected = int(
                row[column]
            )

            if observed != expected:

                add_error(
                    errors,
                    sample_id,
                    "pixel_count",
                    (
                        f"{CLASS_NAMES[class_id]}: "
                        f"observed={observed}, "
                        f"manifest={expected}"
                    ),
                )

        # ----------------------------------------------------
        # INSTANCE COUNTS
        # ----------------------------------------------------

        observed_instances = {}

        for class_id in (1, 2, 3):

            observed = count_instances(
                instance,
                class_id,
            )

            observed_instances[
                class_id
            ] = observed

            column = INSTANCE_COLUMNS[
                class_id
            ]

            expected = int(
                row[column]
            )

            if observed != expected:

                add_error(
                    errors,
                    sample_id,
                    "instance_count",
                    (
                        f"{CLASS_NAMES[class_id]}: "
                        f"observed={observed}, "
                        f"manifest={expected}"
                    ),
                )

        # ----------------------------------------------------
        # SAMPLE STATUS
        # ----------------------------------------------------

        sample_error_count = (
            len(errors)
            - sample_errors_before
        )

        status = (
            "PASS"
            if sample_error_count == 0
            else "FAIL"
        )

        sample_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "city": row["city"],

                "height": height,
                "width": width,

                "person_pixels":
                    observed_pixels[1],

                "car_pixels":
                    observed_pixels[2],

                "motorcycle_pixels":
                    observed_pixels[3],

                "person_instances":
                    observed_instances[1],

                "car_instances":
                    observed_instances[2],

                "motorcycle_instances":
                    observed_instances[3],

                "target_values":
                    ",".join(
                        map(
                            str,
                            sorted(
                                unique_target_values
                            ),
                        )
                    ),

                "status": status,
            }
        )

    # ========================================================
    # SAMPLE SUMMARY CSV
    # ========================================================

    sample_summary = pd.DataFrame(
        sample_rows
    )

    sample_summary.to_csv(
        sample_summary_path,
        index=False,
    )

    # ========================================================
    # ERROR CSV
    # ========================================================

    errors_df = pd.DataFrame(
        errors,
        columns=[
            "sample_id",
            "error_type",
            "message",
        ],
    )

    errors_df.to_csv(
        errors_path,
        index=False,
    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    summary_rows = []

    for split_name in [
        "train",
        "val",
        "total",
    ]:

        if split_name == "total":

            split_df = manifest

        else:

            split_df = manifest[
                manifest["split"]
                == split_name
            ]

        summary_rows.append(
            {
                "split": split_name,

                "samples":
                    len(split_df),

                "person_pixels":
                    int(
                        split_df[
                            "person_pixels"
                        ].sum()
                    ),

                "car_pixels":
                    int(
                        split_df[
                            "car_pixels"
                        ].sum()
                    ),

                "motorcycle_pixels":
                    int(
                        split_df[
                            "motorcycle_pixels"
                        ].sum()
                    ),

                "person_instances":
                    int(
                        split_df[
                            "person_instances"
                        ].sum()
                    ),

                "car_instances":
                    int(
                        split_df[
                            "car_instances"
                        ].sum()
                    ),

                "motorcycle_instances":
                    int(
                        split_df[
                            "motorcycle_instances"
                        ].sum()
                    ),
            }
        )

    dataset_summary = pd.DataFrame(
        summary_rows
    )

    dataset_summary.to_csv(
        dataset_summary_path,
        index=False,
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    passed_samples = int(
        (
            sample_summary["status"]
            == "PASS"
        ).sum()
    )

    failed_samples = int(
        (
            sample_summary["status"]
            == "FAIL"
        ).sum()
    )

    total_person_pixels = int(
        manifest[
            "person_pixels"
        ].sum()
    )

    total_car_pixels = int(
        manifest[
            "car_pixels"
        ].sum()
    )

    total_motorcycle_pixels = int(
        manifest[
            "motorcycle_pixels"
        ].sum()
    )

    total_person_instances = int(
        manifest[
            "person_instances"
        ].sum()
    )

    total_car_instances = int(
        manifest[
            "car_instances"
        ].sum()
    )

    total_motorcycle_instances = int(
        manifest[
            "motorcycle_instances"
        ].sum()
    )

    final_status = (
        "PASSED"
        if len(errors) == 0
        else "FAILED"
    )

    # ========================================================
    # TEXT REPORT
    # ========================================================

    report = f"""
STEP 01: DATASET VALIDATION
========================================================================

Dataset root:
{DATASET_ROOT}

Manifest:
{MANIFEST_PATH}


DATASET SPLIT
------------------------------------------------------------------------

Total samples       : {total_samples}
Train samples       : {train_samples}
Validation samples  : {val_samples}


IMAGE INFORMATION
------------------------------------------------------------------------

Expected resolution : {EXPECTED_WIDTH} x {EXPECTED_HEIGHT}
Channels            : 3 (RGB)


TARGET CLASS MAPPING
------------------------------------------------------------------------

0 = Background
1 = Person (Cityscapes Person + Rider)
2 = Car
3 = Motorcycle


PIXEL COUNTS
------------------------------------------------------------------------

Person pixels       : {total_person_pixels:,}
Car pixels          : {total_car_pixels:,}
Motorcycle pixels   : {total_motorcycle_pixels:,}


INSTANCE COUNTS
------------------------------------------------------------------------

Person instances    : {total_person_instances:,}
Car instances       : {total_car_instances:,}
Motorcycle instances: {total_motorcycle_instances:,}


VALIDATION RESULTS
------------------------------------------------------------------------

Passed samples      : {passed_samples}
Failed samples      : {failed_samples}
Validation errors   : {len(errors)}

FINAL STATUS        : {final_status}
""".strip()

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print()
    print("=" * 72)

    print(
        f"Samples      : "
        f"{total_samples}"
    )

    print(
        f"Train        : "
        f"{train_samples}"
    )

    print(
        f"Validation   : "
        f"{val_samples}"
    )

    print()

    print(
        f"Person pixels       : "
        f"{total_person_pixels:,}"
    )

    print(
        f"Car pixels          : "
        f"{total_car_pixels:,}"
    )

    print(
        f"Motorcycle pixels   : "
        f"{total_motorcycle_pixels:,}"
    )

    print()

    print(
        f"Person instances    : "
        f"{total_person_instances:,}"
    )

    print(
        f"Car instances       : "
        f"{total_car_instances:,}"
    )

    print(
        f"Motorcycle instances: "
        f"{total_motorcycle_instances:,}"
    )

    print()
    print(
        f"FINAL STATUS: "
        f"{final_status}"
    )

    print()
    print(
        "Outputs saved to:"
    )

    print(output_dir)

    print()
    print("Created:")

    print(
        f"  {report_path.name}"
    )

    print(
        f"  {dataset_summary_path.name}"
    )

    print(
        f"  {sample_summary_path.name}"
    )

    print(
        f"  {errors_path.name}"
    )

    print("=" * 72)

    if final_status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
    