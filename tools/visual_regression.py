"""Run reproducible visual A/B experiments without modifying the source tree.

Each conversion is launched through Hydra's ``-m`` entry point.  The basic
Hydra sweeper cannot zip ``width`` and ``height``, so this runner invokes one
single-job multirun per required square size instead of accidentally testing a
3×3 rectangular matrix.

The two P2 candidates run from a throw-away package overlay.  This gives the
experiment its requested implementation while keeping the developer's worktree
and committed source untouched.  Patches are exact and deliberately fail when
``perceive.py`` changes, forcing the experiment contract to be reviewed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIXEL_SOURCES = (
    Path("tests/cup.png"),
    Path("tests/cup_no_padding.png"),
    Path("docs/assets/banana_orig.png"),
    Path("docs/assets/chair_orig.png"),
)
SIZES = (48, 64, 96)
ASCII_ROWS = (40, 60)


@dataclass(frozen=True)
class Variant:
    """A named implementation used as one side of an A/B experiment."""

    name: str
    overrides: tuple[str, ...] = ()
    patch: str | None = None


EXPERIMENTS: dict[str, tuple[Variant, Variant]] = {
    "ab-2": (
        Variant("a-bilateral"),
        Variant("b-mean-shift-only", patch="skip-bilateral"),
    ),
    "ab-3": (
        Variant("a-fixed-canny", ("perceive.canny_low=40", "perceive.canny_high=120")),
        Variant("b-otsu-canny", patch="otsu-canny"),
    ),
}


def _source_slug(source: Path) -> str:
    return source.stem.replace("_orig", "")


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(
            f"experimental patch is stale: expected one occurrence in {path}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def _apply_patch(package_dir: Path, patch: str) -> None:
    perceive = package_dir / "perceive.py"
    if patch == "skip-bilateral":
        _replace_once(
            perceive,
            "denoised = cv2.bilateralFilter(bgr, denoise_d, denoise_sigma, denoise_sigma)",
            "denoised = bgr.copy()  # AB-2 temporary candidate: skip bilateral filtering",
        )
    elif patch == "otsu-canny":
        _replace_once(
            perceive,
            "canny = cv2.Canny(gray, canny_low, canny_high)",
            """otsu, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Match the current ASCII photo-edge policy: ratio × Otsu, with a floor.
    canny = cv2.Canny(gray, max(float(otsu) * 0.33, 10.0), float(otsu))""",
        )
    else:
        raise ValueError(f"unknown experimental patch: {patch}")


@contextmanager
def _overlay_for(variant: Variant) -> Iterator[Path | None]:
    """Yield a temporary import root containing a patched package if needed."""
    if variant.patch is None:
        yield None
        return

    with tempfile.TemporaryDirectory(prefix="img2pixelart-ab-") as tmp:
        root = Path(tmp)
        package = root / "img2pixelart"
        shutil.copytree(
            ROOT / "img2pixelart",
            package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        _apply_patch(package, variant.patch)
        yield root


def _run(command: list[str], overlay: Path | None, cwd: Path) -> None:
    env = os.environ.copy()
    if overlay is not None:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(overlay) if not existing else f"{overlay}{os.pathsep}{existing}"
        )
    # A neutral cwd keeps the repo's own `img2pixelart/` directory from
    # shadowing the overlay (sys.path[0] would otherwise be the repo root).
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _uv_run(module_args: list[str]) -> list[str]:
    return ["uv", "run", "--project", str(ROOT), "python", "-m", *module_args]


def _hydra_pixel_command(
    source: Path, size: int, output_dir: Path, variant: Variant
) -> list[str]:
    return _uv_run(
        [
            "img2pixelart.cli",
            "-m",
            f"img={source.resolve()}",
            f"width={size}",
            f"height={size}",
            *variant.overrides,
            f"hydra.sweep.dir={output_dir.resolve()}",
            "hydra.sweep.subdir=${hydra:job.override_dirname}",
        ]
    )


def _find_result(directory: Path) -> Path:
    results = sorted(directory.rglob("result.png"))
    if len(results) != 1:
        raise RuntimeError(
            f"expected exactly one result.png under {directory}, found {len(results)}"
        )
    return results[0]


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot read generated image: {path}")
    return image


def _image_metrics(a_path: Path, b_path: Path) -> dict[str, object]:
    a, b = _read_image(a_path), _read_image(b_path)
    if a.shape != b.shape:
        raise RuntimeError(f"image shapes differ: {a.shape} != {b.shape}")
    difference = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "pixels": int(a.shape[0] * a.shape[1]),
        "changed_pixel_ratio": float(np.any(difference != 0, axis=-1).mean()),
        "mean_absolute_channel_difference": float(difference.mean()),
        "maximum_channel_difference": int(difference.max()),
    }


def _strip_metrics(a_result: Path, b_result: Path, name: str) -> dict[str, object]:
    a_strip, b_strip = a_result.parent / name, b_result.parent / name
    if not a_strip.is_file() or not b_strip.is_file():
        return {"available": False}
    a, b = _read_image(a_strip), _read_image(b_strip)
    if a.shape != b.shape:
        return {
            "available": True,
            "shape_equal": False,
            "identical": False,
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
        }
    return {
        "available": True,
        "shape_equal": True,
        "identical": _image_metrics(a_strip, b_strip)["changed_pixel_ratio"] == 0.0,
    }


def _comparison_sheet(rows: list[tuple[str, Path, Path]], destination: Path) -> None:
    """Write a labelled nearest-neighbour A/B sheet for visual review."""
    scale, label_height, gutter = 4, 30, 8
    rendered: list[np.ndarray] = []
    for label, a_path, b_path in rows:
        a, b = _read_image(a_path), _read_image(b_path)
        if a.shape[-1] == 4:
            a, b = a[..., :3], b[..., :3]
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        panel = np.zeros(
            (
                label_height + max(a.shape[0], b.shape[0]),
                a.shape[1] + gutter + b.shape[1],
                3,
            ),
            dtype=np.uint8,
        )
        cv2.putText(
            panel,
            f"{label}: A | B",
            (2, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        panel[label_height : label_height + a.shape[0], : a.shape[1]] = a
        panel[label_height : label_height + b.shape[0], a.shape[1] + gutter :] = b
        rendered.append(panel)

    width = max(panel.shape[1] for panel in rendered)
    padded = [
        cv2.copyMakeBorder(panel, 0, 0, 0, width - panel.shape[1], cv2.BORDER_CONSTANT)
        for panel in rendered
    ]
    sheet = np.vstack(padded)
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError(f"failed to write comparison sheet: {destination}")


def run_experiment(name: str, destination: Path) -> Path:
    """Run one specified A/B experiment and save evidence plus metrics."""
    variant_a, variant_b = EXPERIMENTS[name]
    destination.mkdir(parents=True, exist_ok=False)
    pairs: list[tuple[str, Path, Path]] = []
    measurements: dict[str, object] = {}

    for source in PIXEL_SOURCES:
        if not (ROOT / source).is_file():
            raise FileNotFoundError(f"baseline source is missing: {source}")
        for size in SIZES:
            result_paths: list[Path] = []
            for variant in (variant_a, variant_b):
                run_dir = destination / _source_slug(source) / variant.name / str(size)
                with _overlay_for(variant) as overlay:
                    _run(
                        _hydra_pixel_command(source, size, run_dir, variant),
                        overlay,
                        cwd=destination,
                    )
                result_paths.append(_find_result(run_dir))

            a_result, b_result = result_paths
            key = f"{source.as_posix()}@{size}"
            try:
                measurements[key] = {
                    "result": _image_metrics(a_result, b_result),
                    "05_palette": _strip_metrics(a_result, b_result, "05_palette.png"),
                    "22_palette_strip": _strip_metrics(
                        a_result, b_result, "22_palette_strip.png"
                    ),
                    "a": str(a_result.relative_to(destination)),
                    "b": str(b_result.relative_to(destination)),
                }
            except RuntimeError as error:  # keep the run alive for later pairs
                measurements[key] = {
                    "error": f"{type(error).__name__}: {error}",
                    "a": str(a_result.relative_to(destination)),
                    "b": str(b_result.relative_to(destination)),
                }
            pairs.append((key, a_result, b_result))

    _comparison_sheet(pairs, destination / "comparison.png")
    (destination / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": name,
                "variants": [variant_a.name, variant_b.name],
                "source_images": [str(source) for source in PIXEL_SOURCES],
                "sizes": SIZES,
                "measurements": measurements,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def run_baseline(destination: Path) -> Path:
    """Run the issue's image and ASCII matrices with the current implementation."""
    destination.mkdir(parents=True, exist_ok=False)
    baseline = Variant("current")
    for source in PIXEL_SOURCES:
        for size in SIZES:
            run_dir = destination / "pixel" / _source_slug(source) / str(size)
            _run(
                _hydra_pixel_command(source, size, run_dir, baseline),
                None,
                cwd=destination,
            )

        for rows in ASCII_ROWS:
            run_dir = destination / "ascii" / _source_slug(source) / str(rows)
            command = [
                *_uv_run(
                    [
                        "img2pixelart.cli",
                        "ascii",
                        "-m",
                        f"img={source.resolve()}",
                        f"ascii.rows={rows}",
                        f"hydra.sweep.dir={run_dir.resolve()}",
                        "hydra.sweep.subdir=${hydra:job.override_dirname}",
                    ]
                )
            ]
            _run(command, None, cwd=destination)
    return destination


def _destination(topic: str, date: str | None) -> Path:
    stamp = date or datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    return ROOT / "outputs" / "ab" / topic / stamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", choices=("baseline", *EXPERIMENTS))
    parser.add_argument(
        "--date", help="UTC artifact suffix; defaults to the current timestamp"
    )
    args = parser.parse_args()

    destination = _destination(args.topic, args.date)
    result = (
        run_baseline(destination)
        if args.topic == "baseline"
        else run_experiment(args.topic, destination)
    )
    print(result)


if __name__ == "__main__":
    main()
