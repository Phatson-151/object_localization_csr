import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    MANIFEST_PATH,
    STEP02_OUTPUT,
    TIKHONOV_LAMBDA,
    TIKHONOV_PADDING,
)

from src.data.dataset_utils import (
    load_sample,
)

from src.preprocessing.image_preprocessing import (
    preprocess_image,
    validate_preprocessed,
)

from src.utils.output_utils import (
    create_output_dir,
    step_filename,
    sample_filename,
)


STEP_NUMBER = 2


def add_error(
    errors: list,
    sample_id: str,
    error_type: str,
    message: str,
):
    """
    Store one preprocessing error.
    """

    errors.append(
        {
            "sample_id": sample_id,
            "error_type": error_type,
            "message": message,
        }
    )


def save_comparison_figure(
    image: np.ndarray,
    grayscale: np.ndarray,
    lowpass: np.ndarray,
    highpass: np.ndarray,
    target: np.ndarray,
    output_path,
    sample_id: str,
):
    """
    Save visual comparison of preprocessing
    stages for one sample.
    """

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 8),
    )

    # --------------------------------------------------------
    # ORIGINAL RGB
    # --------------------------------------------------------

    axes[0, 0].imshow(
        image
    )

    axes[0, 0].set_title(
        "Original RGB"
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    axes[0, 1].imshow(
        grayscale,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    axes[0, 1].set_title(
        "Grayscale"
    )

    # --------------------------------------------------------
    # LOW-PASS
    # --------------------------------------------------------

    axes[0, 2].imshow(
        lowpass,
        cmap="gray",
    )

    axes[0, 2].set_title(
        "Tikhonov Low-pass"
    )

    # --------------------------------------------------------
    # HIGH-PASS
    # --------------------------------------------------------

    highpass_absmax = float(
        np.max(
            np.abs(highpass)
        )
    )

    if highpass_absmax == 0:
        highpass_absmax = 1.0

    axes[1, 0].imshow(
        highpass,
        cmap="gray",
        vmin=-highpass_absmax,
        vmax=highpass_absmax,
    )

    axes[1, 0].set_title(
        "High-pass (CSC Input)"
    )

    # --------------------------------------------------------
    # TARGET MASK
    # --------------------------------------------------------

    target_image = axes[1, 1].imshow(
        target,
        cmap="tab10",
        vmin=0,
        vmax=3,
    )

    axes[1, 1].set_title(
        "Target Mask"
    )

    colorbar = fig.colorbar(
        target_image,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04,
        ticks=[0, 1, 2, 3],
    )

    colorbar.ax.set_yticklabels(
        [
            "Background",
            "Person",
            "Car",
            "Motorcycle",
        ]
    )

    # --------------------------------------------------------
    # HIGH-PASS MAGNITUDE
    # --------------------------------------------------------

    axes[1, 2].imshow(
        np.abs(highpass),
        cmap="gray",
    )

    axes[1, 2].set_title(
        "|High-pass|"
    )

    # --------------------------------------------------------
    # REMOVE AXES
    # --------------------------------------------------------

    for ax in axes.flat:
        ax.axis("off")

    # Colorbar axis needs to remain visible
    colorbar.ax.set_visible(True)

    fig.suptitle(
        f"STEP 02 Preprocessing — {sample_id}",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    output_dir = create_output_dir(
        STEP02_OUTPUT
    )

    array_dir = create_output_dir(
        output_dir / "arrays"
    )

    visualization_dir = (
        create_output_dir(
            output_dir
            / "visualizations"
        )
    )

    report_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "preprocessing_report",
            "txt",
        )
    )

    summary_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "preprocessing_summary",
            "csv",
        )
    )

    errors_path = (
        output_dir
        / step_filename(
            STEP_NUMBER,
            "preprocessing_errors",
            "csv",
        )
    )

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    if not MANIFEST_PATH.exists():

        message = (
            f"Manifest not found: "
            f"{MANIFEST_PATH}"
        )

        report_path.write_text(
            message,
            encoding="utf-8",
        )

        print(message)

        raise SystemExit(1)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    total_samples = len(
        manifest
    )

    errors = []

    summary_rows = []

    print("=" * 72)
    print("STEP 02: IMAGE PREPROCESSING")
    print("=" * 72)

    print(
        f"Tikhonov lambda  : "
        f"{TIKHONOV_LAMBDA}"
    )

    print(
        f"Tikhonov padding : "
        f"{TIKHONOV_PADDING}"
    )

    print()

    # ========================================================
    # PROCESS EVERY SAMPLE
    # ========================================================

    for index, row in (
        manifest.iterrows()
    ):

        split = str(
            row["split"]
        )

        sample_id = str(
            row["sample_id"]
        )

        sample_error_start = (
            len(errors)
        )

        try:

            sample = load_sample(
                split=split,
                sample_id=sample_id,
            )

            image = sample[
                "image"
            ]

            target = sample[
                "target"
            ]

            result = preprocess_image(
                image=image,
                lmbda=TIKHONOV_LAMBDA,
                padding=TIKHONOV_PADDING,
            )

            grayscale = result[
                "grayscale"
            ]

            lowpass = result[
                "lowpass"
            ]

            highpass = result[
                "highpass"
            ]

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            preprocessing_errors = (
                validate_preprocessed(
                    original_image=image,
                    grayscale=grayscale,
                    lowpass=lowpass,
                    highpass=highpass,
                    target_mask=target,
                )
            )

            for message in (
                preprocessing_errors
            ):

                add_error(
                    errors,
                    sample_id,
                    "preprocessing_validation",
                    message,
                )

            # ------------------------------------------------
            # SAVE HIGH-PASS ARRAY
            # ------------------------------------------------

            array_name = (
                sample_filename(
                    step=STEP_NUMBER,
                    sample_id=sample_id,
                    name="highpass",
                    extension="npy",
                )
            )

            array_path = (
                array_dir
                / array_name
            )

            np.save(
                array_path,
                highpass.astype(
                    np.float32
                ),
                allow_pickle=False,
            )

            # ------------------------------------------------
            # VERIFY SAVED ARRAY
            # ------------------------------------------------

            saved_highpass = np.load(
                array_path,
                allow_pickle=False,
            )

            if (
                saved_highpass.shape
                != highpass.shape
            ):

                add_error(
                    errors,
                    sample_id,
                    "saved_array_shape",
                    (
                        f"Saved array shape "
                        f"{saved_highpass.shape} "
                        f"!= expected "
                        f"{highpass.shape}"
                    ),
                )

            if (
                saved_highpass.dtype
                != np.float32
            ):

                add_error(
                    errors,
                    sample_id,
                    "saved_array_dtype",
                    (
                        f"Saved dtype "
                        f"{saved_highpass.dtype} "
                        "!= float32"
                    ),
                )

            # ------------------------------------------------
            # SAVE VISUALIZATION
            # ------------------------------------------------

            figure_name = (
                sample_filename(
                    step=STEP_NUMBER,
                    sample_id=sample_id,
                    name="comparison",
                    extension="png",
                )
            )

            figure_path = (
                visualization_dir
                / figure_name
            )

            save_comparison_figure(
                image=image,
                grayscale=grayscale,
                lowpass=lowpass,
                highpass=highpass,
                target=target,
                output_path=figure_path,
                sample_id=sample_id,
            )

            # ------------------------------------------------
            # STATS
            # ------------------------------------------------

            sample_error_count = (
                len(errors)
                - sample_error_start
            )

            status = (
                "PASS"
                if sample_error_count == 0
                else "FAIL"
            )

            summary_rows.append(
                {
                    "sample_id":
                        sample_id,

                    "split":
                        split,

                    "height":
                        highpass.shape[0],

                    "width":
                        highpass.shape[1],

                    "grayscale_min":
                        float(
                            grayscale.min()
                        ),

                    "grayscale_max":
                        float(
                            grayscale.max()
                        ),

                    "grayscale_mean":
                        float(
                            grayscale.mean()
                        ),

                    "grayscale_std":
                        float(
                            grayscale.std()
                        ),

                    "highpass_min":
                        float(
                            highpass.min()
                        ),

                    "highpass_max":
                        float(
                            highpass.max()
                        ),

                    "highpass_mean":
                        float(
                            highpass.mean()
                        ),

                    "highpass_std":
                        float(
                            highpass.std()
                        ),

                    "highpass_abs_mean":
                        float(
                            np.abs(
                                highpass
                            ).mean()
                        ),

                    "highpass_dtype":
                        str(
                            highpass.dtype
                        ),

                    "highpass_file":
                        str(array_path),

                    "visualization_file":
                        str(figure_path),

                    "status":
                        status,
                }
            )

            print(
                f"[{index + 1:02d}/"
                f"{total_samples:02d}] "
                f"{split:<5} | "
                f"{sample_id} | "
                f"std={highpass.std():.6f} | "
                f"{status}"
            )

        except Exception as exc:

            add_error(
                errors,
                sample_id,
                "processing_error",
                repr(exc),
            )

            summary_rows.append(
                {
                    "sample_id":
                        sample_id,

                    "split":
                        split,

                    "height":
                        None,

                    "width":
                        None,

                    "grayscale_min":
                        None,

                    "grayscale_max":
                        None,

                    "grayscale_mean":
                        None,

                    "grayscale_std":
                        None,

                    "highpass_min":
                        None,

                    "highpass_max":
                        None,

                    "highpass_mean":
                        None,

                    "highpass_std":
                        None,

                    "highpass_abs_mean":
                        None,

                    "highpass_dtype":
                        None,

                    "highpass_file":
                        None,

                    "visualization_file":
                        None,

                    "status":
                        "FAIL",
                }
            )

            print(
                f"[{index + 1:02d}/"
                f"{total_samples:02d}] "
                f"{split:<5} | "
                f"{sample_id} | "
                "FAIL"
            )

            print(
                f"    {exc}"
            )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # SAVE ERRORS
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
    # FINAL STATISTICS
    # ========================================================

    passed = int(
        (
            summary_df["status"]
            == "PASS"
        ).sum()
    )

    failed = int(
        (
            summary_df["status"]
            == "FAIL"
        ).sum()
    )

    if passed > 0:

        successful_df = (
            summary_df[
                summary_df[
                    "status"
                ]
                == "PASS"
            ]
        )

        mean_highpass_std = float(
            successful_df[
                "highpass_std"
            ].mean()
        )

        mean_highpass_abs = float(
            successful_df[
                "highpass_abs_mean"
            ].mean()
        )

    else:

        mean_highpass_std = (
            float("nan")
        )

        mean_highpass_abs = (
            float("nan")
        )

    final_status = (
        "PASSED"
        if (
            failed == 0
            and len(errors) == 0
        )
        else "FAILED"
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = f"""
STEP 02: IMAGE PREPROCESSING
========================================================================

INPUT
------------------------------------------------------------------------

Total samples       : {total_samples}

Input format        : RGB
Input size          : 1024 x 2048 x 3


PREPROCESSING PIPELINE
------------------------------------------------------------------------

1. RGB uint8 -> RGB float32 [0,1]
2. RGB -> Grayscale
3. Tikhonov low-pass / high-pass decomposition
4. Save high-pass component as float32 NPY

No resize, crop, or geometric transformation is performed.


PARAMETERS
------------------------------------------------------------------------

Tikhonov lambda     : {TIKHONOV_LAMBDA}
Tikhonov padding    : {TIKHONOV_PADDING}


OUTPUT
------------------------------------------------------------------------

High-pass size      : 1024 x 2048
High-pass dtype     : float32

Mean high-pass std  : {mean_highpass_std:.8f}
Mean |high-pass|    : {mean_highpass_abs:.8f}


VALIDATION
------------------------------------------------------------------------

Passed samples      : {passed}
Failed samples      : {failed}
Errors              : {len(errors)}

FINAL STATUS        : {final_status}


NOTE
------------------------------------------------------------------------

High-pass preprocessing is a project implementation choice following
common SPORCO convolutional sparse representation practice. It is not
a mandatory preprocessing step specified by the SCSC paper.

The high-pass arrays generated in this step will be used as the
convolutional sparse representation input in later CSC / SCSC steps.
""".strip()

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print("=" * 72)

    print(
        f"Processed samples : "
        f"{total_samples}"
    )

    print(
        f"Passed            : "
        f"{passed}"
    )

    print(
        f"Failed            : "
        f"{failed}"
    )

    print(
        f"Errors            : "
        f"{len(errors)}"
    )

    print()

    print(
        f"Mean highpass std : "
        f"{mean_highpass_std:.8f}"
    )

    print(
        f"Mean |highpass|   : "
        f"{mean_highpass_abs:.8f}"
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

    print(
        output_dir
    )

    print()

    print("Created:")

    print(
        f"  {report_path.name}"
    )

    print(
        f"  {summary_path.name}"
    )

    print(
        f"  {errors_path.name}"
    )

    print(
        f"  arrays/"
    )

    print(
        f"  visualizations/"
    )

    print("=" * 72)

    if final_status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
    