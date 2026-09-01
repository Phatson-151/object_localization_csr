import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd

from sporco.dictlrn import cbpdndl

from src.config import (
    MANIFEST_PATH,
    STEP04_OUTPUT,

    CSC_NUM_TRAIN_SAMPLES,
    CSC_FILTER_SIZE,
    CSC_NUM_FILTERS,
    CSC_TRAIN_SCALE,
    CSC_OUTER_ITER,
    CSC_RANDOM_SEED,

    CSC_PAPER_BETA_REFERENCE,
    CSC_LAMBDA_RATIO,
    CSC_LAMBDA_MAX_SAFETY,

    CSC_CMOD_RHO,

    CSC_DEGENERATE_L1_TOL,
    CSC_MIN_OBJECTIVE_REL_CHANGE,
    CSC_MIN_DICTIONARY_REL_CHANGE,
)

from src.csc.csc_utils import (
    build_training_stack,
    initialize_dictionary,
    squeeze_dictionary,
    dictionary_statistics,
    iteration_stats_to_dataframe,
    save_dictionary_grid,
    save_dictionary_comparison,
    save_training_curves,
    estimate_cbpdn_lambda_max,
)

from src.utils.output_utils import (
    create_output_dir,
)


STEP_NUMBER = 4


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "STEP 04: Train full unsupervised "
            "CSC baseline using SPORCO."
        )
    )

    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=CSC_NUM_TRAIN_SAMPLES,
    )

    parser.add_argument(
        "--num-filters",
        type=int,
        default=CSC_NUM_FILTERS,
    )

    parser.add_argument(
        "--filter-size",
        type=int,
        default=CSC_FILTER_SIZE,
    )

    # --------------------------------------------------------
    # FIXED LAMBDA
    #
    # Default is None.
    #
    # If omitted:
    #   lambda = lambda_ratio * lambda_max
    #
    # Only use --lambda when intentionally testing
    # an absolute value.
    # --------------------------------------------------------

    parser.add_argument(
        "--lambda",
        dest="fixed_lambda",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--lambda-ratio",
        type=float,
        default=CSC_LAMBDA_RATIO,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=CSC_TRAIN_SCALE,
    )

    parser.add_argument(
        "--outer-iter",
        type=int,
        default=CSC_OUTER_ITER,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CSC_RANDOM_SEED,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# SMALL HELPERS
# ============================================================

def get_log_value(
    log_df,
    column,
    position,
    default=np.nan,
):

    if (
        column not in log_df.columns
        or len(log_df) == 0
    ):

        return float(default)

    return float(
        log_df[
            column
        ].iloc[
            position
        ]
    )


def relative_dictionary_change(
    initial_dictionary,
    final_dictionary,
):

    D0 = squeeze_dictionary(
        initial_dictionary
    ).astype(
        np.float64
    )

    D1 = squeeze_dictionary(
        final_dictionary
    ).astype(
        np.float64
    )

    if D0.shape != D1.shape:

        return float("inf")

    denominator = float(
        np.linalg.norm(
            D0
        )
    )

    denominator = max(
        denominator,
        1e-12,
    )

    return float(
        np.linalg.norm(
            D1 - D0
        )
        /
        denominator
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # ARGUMENT VALIDATION
    # ========================================================

    if args.lambda_ratio <= 0.0:

        raise ValueError(
            "--lambda-ratio must be > 0"
        )

    if args.lambda_ratio >= 1.0:

        raise ValueError(
            "--lambda-ratio must be < 1"
        )

    if (
        args.fixed_lambda is not None
        and args.fixed_lambda <= 0.0
    ):

        raise ValueError(
            "--lambda must be > 0"
        )

    # ========================================================
    # RUN NAME
    # ========================================================

    if args.run_name is None:

        run_name = (
            f"csc_k{args.num_filters}_"
            f"n{args.num_train_samples}_"
            f"s{args.scale:g}_"
            f"lr{args.lambda_ratio:g}"
        )

    else:

        run_name = (
            args.run_name
        )

    root_output = create_output_dir(
        STEP04_OUTPUT
    )

    run_output = (
        root_output
        / run_name
    )

    if run_output.exists():

        if args.overwrite:

            shutil.rmtree(
                run_output
            )

        elif any(
            run_output.iterdir()
        ):

            raise FileExistsError(
                f"Run already exists:\n"
                f"{run_output}\n"
                f"Use --overwrite."
            )

    create_output_dir(
        run_output
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    config_path = (
        run_output
        / "step04_run_config.json"
    )

    lambda_path = (
        run_output
        / "step04_lambda_diagnostics.json"
    )

    training_samples_path = (
        run_output
        / "step04_training_samples.csv"
    )

    initial_dictionary_path = (
        run_output
        / "step04_initial_dictionary.npy"
    )

    final_dictionary_path = (
        run_output
        / "step04_final_dictionary.npy"
    )

    initial_dictionary_png = (
        run_output
        / "step04_initial_dictionary.png"
    )

    final_dictionary_png = (
        run_output
        / "step04_final_dictionary.png"
    )

    comparison_png = (
        run_output
        / "step04_dictionary_comparison.png"
    )

    log_path = (
        run_output
        / "step04_training_log.csv"
    )

    curves_path = (
        run_output
        / "step04_training_curves.png"
    )

    report_path = (
        run_output
        / "step04_training_report.txt"
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    train_manifest = (
        manifest[
            manifest["split"]
            == "train"
        ]
        .reset_index(
            drop=True
        )
    )

    if (
        args.num_train_samples
        > len(train_manifest)
    ):

        raise ValueError(
            f"Requested "
            f"{args.num_train_samples} "
            f"training images, "
            f"but only "
            f"{len(train_manifest)} exist."
        )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 72)

    print(
        "STEP 04: UNSUPERVISED CSC BASELINE"
    )

    print("=" * 72)

    print(
        f"Run name          : "
        f"{run_name}"
    )

    print(
        f"Training images   : "
        f"{args.num_train_samples}"
    )

    print(
        f"Scale             : "
        f"{args.scale}"
    )

    print(
        f"Filter size       : "
        f"{args.filter_size} x "
        f"{args.filter_size}"
    )

    print(
        f"Number of filters : "
        f"{args.num_filters}"
    )

    if args.fixed_lambda is None:

        print(
            "Lambda mode       : "
            "DATA-DRIVEN"
        )

        print(
            f"Lambda ratio      : "
            f"{args.lambda_ratio}"
        )

    else:

        print(
            "Lambda mode       : "
            "FIXED"
        )

        print(
            f"Requested lambda  : "
            f"{args.fixed_lambda}"
        )

    print(
        f"Outer iterations  : "
        f"{args.outer_iter}"
    )

    print(
        f"Random seed       : "
        f"{args.seed}"
    )

    print()

    # ========================================================
    # TRAINING SIGNALS
    # ========================================================

    print(
        "Loading STEP 02 high-pass signals..."
    )

    S, training_metadata = (
        build_training_stack(
            train_manifest=
                train_manifest,

            num_samples=
                args.num_train_samples,

            scale=
                args.scale,
        )
    )

    training_metadata.to_csv(
        training_samples_path,
        index=False,
    )

    height = int(
        S.shape[0]
    )

    width = int(
        S.shape[1]
    )

    num_images = int(
        S.shape[2]
    )

    print(
        f"Training tensor   : "
        f"{S.shape}"
    )

    print(
        f"Signal dtype      : "
        f"{S.dtype}"
    )

    coef_entries = (
        height
        * width
        * num_images
        * args.num_filters
    )

    coef_mib = (
        coef_entries
        * 4
        / (1024 ** 2)
    )

    print(
        f"Coefficient bank  : "
        f"~{coef_mib:.1f} MiB"
    )

    print()

    # ========================================================
    # INITIAL DICTIONARY
    # ========================================================

    D0 = initialize_dictionary(
        filter_size=
            args.filter_size,

        num_filters=
            args.num_filters,

        seed=
            args.seed,
    )

    np.save(
        initial_dictionary_path,
        D0,
        allow_pickle=False,
    )

    save_dictionary_grid(
        D0,
        initial_dictionary_png,
        title=(
            "STEP 04 — "
            "Initial CSC Dictionary"
        ),
    )

    # ========================================================
    # ESTIMATE LAMBDA_MAX
    # ========================================================

    print(
        "Estimating lambda_max "
        "from full training data..."
    )

    lambda_max = (
        estimate_cbpdn_lambda_max(
            dictionary=D0,
            signals=S,
        )
    )

    # --------------------------------------------------------
    # EFFECTIVE LAMBDA
    # --------------------------------------------------------

    if args.fixed_lambda is None:

        effective_lambda = (
            args.lambda_ratio
            * lambda_max
        )

        lambda_source = (
            "data_driven"
        )

    else:

        effective_lambda = float(
            args.fixed_lambda
        )

        lambda_source = (
            "fixed"
        )

        if (
            effective_lambda
            >=
            CSC_LAMBDA_MAX_SAFETY
            * lambda_max
        ):

            raise ValueError(
                "\nRequested lambda is too close "
                "to or above lambda_max.\n"
                f"lambda_max       = "
                f"{lambda_max:.8e}\n"
                f"requested lambda = "
                f"{effective_lambda:.8e}\n"
                "\nThis is likely to produce an "
                "all-zero sparse representation."
            )

    print()

    print(
        f"Estimated lambda_max : "
        f"{lambda_max:.8e}"
    )

    print(
        f"Effective lambda     : "
        f"{effective_lambda:.8e}"
    )

    print(
        f"lambda/lambda_max    : "
        f"{effective_lambda / lambda_max:.6f}"
    )

    print(
        f"Paper beta reference : "
        f"{CSC_PAPER_BETA_REFERENCE}"
    )

    print()

    # ========================================================
    # LAMBDA DIAGNOSTICS
    # ========================================================

    lambda_diagnostics = {
        "lambda_mode":
            lambda_source,

        "lambda_max":
            lambda_max,

        "lambda_ratio":
            (
                effective_lambda
                / lambda_max
            ),

        "effective_lambda":
            effective_lambda,

        "paper_beta_reference":
            CSC_PAPER_BETA_REFERENCE,

        "note": (
            "Paper beta=0.5 is retained only "
            "as a reference. Effective lambda "
            "is adapted to the numerical scale "
            "of this project's high-pass signal."
        ),
    }

    lambda_path.write_text(
        json.dumps(
            lambda_diagnostics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # ADMM RHO
    # ========================================================

    rho_x = (
        50.0
        * effective_lambda
        + 0.5
    )

    rho_d = (
        CSC_CMOD_RHO
    )

    print(
        f"CBPDN rho            : "
        f"{rho_x:.8f}"
    )

    print(
        f"Dictionary rho        : "
        f"{rho_d:.8f}"
    )

    # ========================================================
    # SPORCO OPTIONS
    # ========================================================

    options = (
        cbpdndl
        .ConvBPDNDictLearn
        .Options(
            {
                "Verbose":
                    True,

                "MaxMainIter":
                    args.outer_iter,

                "AccurateDFid":
                    True,

                "CBPDN": {
                    "rho":
                        rho_x,

                    "MaxMainIter":
                        1,

                    "AutoRho": {
                        "Enabled":
                            False,
                    },
                },

                "CCMOD": {
                    "rho":
                        rho_d,

                    "MaxMainIter":
                        1,

                    "ZeroMean":
                        True,

                    "AutoRho": {
                        "Enabled":
                            False,
                    },
                },
            },

            xmethod="admm",
            dmethod="cns",
        )
    )

    # ========================================================
    # RUN CONFIG
    # ========================================================

    run_config = {
        "step":
            STEP_NUMBER,

        "run_name":
            run_name,

        "num_train_samples":
            args.num_train_samples,

        "model_height":
            height,

        "model_width":
            width,

        "scale":
            args.scale,

        "filter_size":
            args.filter_size,

        "num_filters":
            args.num_filters,

        "lambda_mode":
            lambda_source,

        "lambda_max":
            lambda_max,

        "lambda_ratio":
            effective_lambda
            / lambda_max,

        "effective_lambda":
            effective_lambda,

        "paper_beta_reference":
            CSC_PAPER_BETA_REFERENCE,

        "outer_iterations":
            args.outer_iter,

        "random_seed":
            args.seed,

        "xmethod":
            "admm",

        "dmethod":
            "cns",

        "cbpdn_rho":
            rho_x,

        "ccmod_rho":
            rho_d,

        "auto_rho":
            False,

        "inner_cbpdn_steps_per_outer":
            1,

        "inner_ccmod_steps_per_outer":
            1,
    }

    config_path.write_text(
        json.dumps(
            run_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()

    print("=" * 72)

    print(
        "SPORCO CONVOLUTIONAL "
        "DICTIONARY LEARNING"
    )

    print("=" * 72)

    solver = (
        cbpdndl.ConvBPDNDictLearn(
            D0,
            S,
            effective_lambda,
            options,
            xmethod="admm",
            dmethod="cns",
            dimK=1,
            dimN=2,
        )
    )

    solver.solve()

    elapsed = float(
        solver.timer.elapsed(
            "solve"
        )
    )

    # ========================================================
    # FINAL DICTIONARY
    # ========================================================

    D1 = solver.getdict(
        crop=True
    )

    D1 = squeeze_dictionary(
        D1
    )

    np.save(
        final_dictionary_path,
        D1.astype(
            np.float32
        ),
        allow_pickle=False,
    )

    # ========================================================
    # ITERATION LOG
    # ========================================================

    iteration_stats = (
        solver.getitstat()
    )

    log_df = (
        iteration_stats_to_dataframe(
            iteration_stats
        )
    )

    log_df.to_csv(
        log_path,
        index=False,
    )

    # ========================================================
    # VISUALIZATIONS
    # ========================================================

    save_dictionary_grid(
        D1,
        final_dictionary_png,
        title=(
            "STEP 04 — "
            "Learned CSC Dictionary"
        ),
    )

    save_dictionary_comparison(
        D0,
        D1,
        comparison_png,
    )

    save_training_curves(
        log_df,
        curves_path,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    dictionary_stats = (
        dictionary_statistics(
            D1
        )
    )

    dictionary_change = (
        relative_dictionary_change(
            D0,
            D1,
        )
    )

    errors = []
    warnings = []

    # --------------------------------------------------------
    # DICTIONARY BASIC CHECKS
    # --------------------------------------------------------

    if not dictionary_stats[
        "finite"
    ]:

        errors.append(
            "Final dictionary contains "
            "NaN or Inf."
        )

    if (
        dictionary_stats[
            "num_filters"
        ]
        != args.num_filters
    ):

        errors.append(
            "Incorrect number of "
            "dictionary filters."
        )

    if (
        D1.shape[0]
        != args.filter_size
        or D1.shape[1]
        != args.filter_size
    ):

        errors.append(
            "Incorrect dictionary "
            "filter size."
        )

    # --------------------------------------------------------
    # L1 CHECK
    #
    # This catches exactly the failure from
    # the previous lambda=0.5 run.
    # --------------------------------------------------------

    if "RegL1" in log_df.columns:

        max_l1 = float(
            log_df[
                "RegL1"
            ].max()
        )

        final_l1 = float(
            log_df[
                "RegL1"
            ].iloc[-1]
        )

        if (
            max_l1
            <= CSC_DEGENERATE_L1_TOL
        ):

            errors.append(
                "Degenerate all-zero sparse "
                "representation detected: "
                "RegL1 remained zero."
            )

    else:

        max_l1 = float("nan")
        final_l1 = float("nan")

        warnings.append(
            "RegL1 was not present in "
            "SPORCO iteration statistics."
        )

    # --------------------------------------------------------
    # OBJECTIVE CHANGE CHECK
    # --------------------------------------------------------

    if (
        "ObjFun" in log_df.columns
        and len(log_df) > 1
    ):

        first_objective = float(
            log_df[
                "ObjFun"
            ].iloc[0]
        )

        final_objective = float(
            log_df[
                "ObjFun"
            ].iloc[-1]
        )

        objective_min = float(
            log_df[
                "ObjFun"
            ].min()
        )

        objective_max = float(
            log_df[
                "ObjFun"
            ].max()
        )

        objective_relative_change = (
            (
                objective_max
                - objective_min
            )
            /
            max(
                abs(
                    first_objective
                ),
                1e-12,
            )
        )

        if (
            objective_relative_change
            <=
            CSC_MIN_OBJECTIVE_REL_CHANGE
        ):

            errors.append(
                "Objective function remained "
                "effectively constant. "
                "CSC learning did not progress."
            )

    else:

        first_objective = float("nan")
        final_objective = float("nan")
        objective_relative_change = float(
            "nan"
        )

        warnings.append(
            "ObjFun was not present in "
            "SPORCO iteration statistics."
        )

    # --------------------------------------------------------
    # DICTIONARY CHANGE CHECK
    # --------------------------------------------------------

    if (
        dictionary_change
        <=
        CSC_MIN_DICTIONARY_REL_CHANGE
    ):

        errors.append(
            "Learned dictionary is "
            "numerically unchanged from "
            "the initial dictionary."
        )

    # --------------------------------------------------------
    # ADMM RESIDUALS
    #
    # Do NOT fail only because residuals are
    # not tiny: this CDL configuration performs
    # one inner ADMM step per outer iteration.
    # We save them as diagnostics instead.
    # --------------------------------------------------------

    final_x_primal = (
        get_log_value(
            log_df,
            "XPrRsdl",
            -1,
        )
    )

    final_x_dual = (
        get_log_value(
            log_df,
            "XDlRsdl",
            -1,
        )
    )

    final_d_primal = (
        get_log_value(
            log_df,
            "DPrRsdl",
            -1,
        )
    )

    final_d_dual = (
        get_log_value(
            log_df,
            "DDlRsdl",
            -1,
        )
    )

    final_dfid = (
        get_log_value(
            log_df,
            "DFid",
            -1,
        )
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    final_status = (
        "PASSED"
        if len(errors) == 0
        else "FAILED"
    )

    # ========================================================
    # REPORT
    # ========================================================

    report_lines = [
        "STEP 04: UNSUPERVISED CSC BASELINE",
        "=" * 72,
        "",
        "RUN CONFIGURATION",
        "-" * 72,
        f"Run name                  : {run_name}",
        f"Training images           : {args.num_train_samples}",
        f"Model resolution          : {width} x {height}",
        f"Scale                     : {args.scale}",
        f"Filter size               : {args.filter_size} x {args.filter_size}",
        f"Number of filters K       : {args.num_filters}",
        f"Outer iterations          : {args.outer_iter}",
        f"Random seed               : {args.seed}",
        "",
        "SPARSITY PARAMETER",
        "-" * 72,
        f"Paper beta reference      : {CSC_PAPER_BETA_REFERENCE}",
        f"Estimated lambda_max      : {lambda_max:.8e}",
        f"lambda / lambda_max       : {effective_lambda / lambda_max:.8f}",
        f"Effective lambda          : {effective_lambda:.8e}",
        f"Lambda mode               : {lambda_source}",
        "",
        "OPTIMIZATION",
        "-" * 72,
        "Sparse coding             : ADMM",
        "Dictionary update         : Consensus ADMM",
        f"CBPDN rho                 : {rho_x:.8e}",
        f"CCMOD rho                 : {rho_d:.8e}",
        "AutoRho                  : Disabled",
        "CBPDN steps / outer       : 1",
        "CCMOD steps / outer       : 1",
        "",
        "TRAINING RESULT",
        "-" * 72,
        f"Solve time (seconds)      : {elapsed:.3f}",
        f"Initial objective         : {first_objective:.8e}",
        f"Final objective           : {final_objective:.8e}",
        f"Objective relative span   : {objective_relative_change:.8e}",
        f"Final data fidelity       : {final_dfid:.8e}",
        f"Maximum L1                : {max_l1:.8e}",
        f"Final L1                  : {final_l1:.8e}",
        f"Dictionary rel. change    : {dictionary_change:.8e}",
        "",
        "FINAL ADMM DIAGNOSTICS",
        "-" * 72,
        f"X primal residual         : {final_x_primal:.8e}",
        f"X dual residual           : {final_x_dual:.8e}",
        f"D primal residual         : {final_d_primal:.8e}",
        f"D dual residual           : {final_d_dual:.8e}",
        "",
        "DICTIONARY",
        "-" * 72,
        f"Final shape               : {D1.shape}",
        f"Filter norm min           : {dictionary_stats['norm_min']:.8f}",
        f"Filter norm max           : {dictionary_stats['norm_max']:.8f}",
        f"Filter norm mean          : {dictionary_stats['norm_mean']:.8f}",
        "",
        "VALIDATION",
        "-" * 72,
        f"Errors                    : {len(errors)}",
        f"Warnings                  : {len(warnings)}",
        f"FINAL STATUS              : {final_status}",
    ]

    if errors:

        report_lines.extend(
            [
                "",
                "ERRORS",
                "-" * 72,
            ]
        )

        for error in errors:

            report_lines.append(
                f"- {error}"
            )

    if warnings:

        report_lines.extend(
            [
                "",
                "WARNINGS",
                "-" * 72,
            ]
        )

        for warning in warnings:

            report_lines.append(
                f"- {warning}"
            )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()

    print("=" * 72)
    print("STEP 04 RESULT")
    print("=" * 72)

    print(
        f"Solve time          : "
        f"{elapsed:.2f} s"
    )

    print(
        f"lambda_max          : "
        f"{lambda_max:.6e}"
    )

    print(
        f"Effective lambda    : "
        f"{effective_lambda:.6e}"
    )

    print(
        f"Initial ObjFun      : "
        f"{first_objective:.6e}"
    )

    print(
        f"Final ObjFun        : "
        f"{final_objective:.6e}"
    )

    print(
        f"Objective rel span  : "
        f"{objective_relative_change:.6e}"
    )

    print(
        f"Maximum L1          : "
        f"{max_l1:.6e}"
    )

    print(
        f"Final L1            : "
        f"{final_l1:.6e}"
    )

    print(
        f"Dictionary change   : "
        f"{dictionary_change:.6e}"
    )

    print(
        f"Dictionary shape    : "
        f"{D1.shape}"
    )

    print(
        f"Filter norm         : "
        f"{dictionary_stats['norm_min']:.6f}"
        f" .. "
        f"{dictionary_stats['norm_max']:.6f}"
    )

    print(
        f"Errors              : "
        f"{len(errors)}"
    )

    print(
        f"Warnings            : "
        f"{len(warnings)}"
    )

    print(
        f"FINAL STATUS        : "
        f"{final_status}"
    )

    print()

    print(
        "Outputs saved to:"
    )

    print(
        run_output
    )

    print()

    print("Created:")

    for path in [
        config_path,
        lambda_path,
        training_samples_path,
        initial_dictionary_path,
        final_dictionary_path,
        initial_dictionary_png,
        final_dictionary_png,
        comparison_png,
        log_path,
        curves_path,
        report_path,
    ]:

        print(
            f"  {path.name}"
        )

    print("=" * 72)

    if final_status == "FAILED":

        sys.exit(1)


if __name__ == "__main__":
    main()