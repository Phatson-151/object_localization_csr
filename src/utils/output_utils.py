from pathlib import Path


def create_output_dir(path: Path) -> Path:
    """
    Create output directory if it does not already exist.
    """

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def step_filename(
    step: int,
    name: str,
    extension: str,
) -> str:
    """
    Create a standardized output filename.

    Example
    -------
    step_filename(
        1,
        "dataset_summary",
        "csv",
    )

    returns
    -------
    step01_dataset_summary.csv
    """

    extension = extension.lstrip(".")

    return (
        f"step{step:02d}_"
        f"{name}."
        f"{extension}"
    )


def sample_filename(
    step: int,
    sample_id: str,
    name: str,
    extension: str,
) -> str:
    """
    Create standardized sample-specific filename.

    Example
    -------
    step01_hamburg_000000_091900_target_mask.png
    """

    extension = extension.lstrip(".")

    return (
        f"step{step:02d}_"
        f"{sample_id}_"
        f"{name}."
        f"{extension}"
    )