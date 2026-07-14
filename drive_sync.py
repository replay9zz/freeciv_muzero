import os
import pathlib
import shutil
import subprocess
import time


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _marker_path(source):
    safe_name = str(pathlib.Path(source).resolve()).replace(os.sep, "_")
    return pathlib.Path(os.getenv("TMPDIR", "/tmp")) / f"freeciv-muzero-drive-sync{safe_name}.stamp"


def _excluded(path):
    parts = path.parts
    if "belief_tensorboard" in parts:
        return True
    if "heatmaps" in parts:
        if "videos" in parts and path.suffix == ".mp4":
            return False
        return True
    if ".tfevents." in path.name and not _env_bool(
        "GOOGLE_DRIVE_RESULTS_INCLUDE_TENSORBOARD", True
    ):
        return True
    return False


def _print_sync_plan(source_path):
    files = []
    if source_path.is_file():
        if not _excluded(source_path):
            files.append(source_path)
    else:
        for path in source_path.rglob("*"):
            if path.is_file() and not _excluded(path):
                files.append(path)
    total = sum(path.stat().st_size for path in files)
    print(f"Drive sync files: {len(files)}", flush=True)
    print(f"Drive sync size: {_human_size(total)}", flush=True)
    for path in files[:50]:
        if source_path.is_file():
            display = path.name
        else:
            try:
                display = path.relative_to(source_path)
            except ValueError:
                display = path
        print(f"  {display}", flush=True)
    if len(files) > 50:
        print(f"  ... {len(files) - 50} more", flush=True)


def _human_size(size):
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024


def _should_sync(source):
    interval = int(os.getenv("GOOGLE_DRIVE_RESULTS_INTERVAL", "300"))
    if interval <= 0:
        return True
    marker = _marker_path(source)
    now = time.time()
    try:
        if now - marker.stat().st_mtime < interval:
            return False
    except FileNotFoundError:
        pass
    marker.touch()
    return True


def sync_path(source, dest_subdir=None, force=False):
    destination = os.getenv("GOOGLE_DRIVE_RESULTS", "").strip()
    if not destination:
        return False

    source_path = pathlib.Path(source).expanduser()
    if not source_path.exists():
        return False
    if not force and not _should_sync(source_path):
        return False

    if dest_subdir is None:
        results_root = pathlib.Path(__file__).resolve().parent / "results"
        try:
            relative = source_path.resolve().relative_to(results_root)
            if str(relative) != ".":
                dest_subdir = str(relative)
        except ValueError:
            pass

    if dest_subdir:
        if destination.endswith(":") or ":" in destination.split(os.sep)[0]:
            destination = f"{destination.rstrip('/')}/{dest_subdir}"
        else:
            destination = str(pathlib.Path(destination).expanduser() / dest_subdir)

    background = _env_bool("GOOGLE_DRIVE_RESULTS_BACKGROUND", True)
    verbose = _env_bool("GOOGLE_DRIVE_RESULTS_VERBOSE", False)
    if verbose:
        _print_sync_plan(source_path)
    if destination.endswith(":") or ":" in destination.split(os.sep)[0]:
        if shutil.which("rclone") is None:
            print("Google Drive sync skipped: rclone not found.", flush=True)
            return False
        cmd = [
            "rclone",
            "copy",
            str(source_path),
            destination,
            "--filter",
            "+ heatmaps/videos/*.mp4",
            "--filter",
            "- heatmaps/**",
            "--filter",
            "- belief_tensorboard/**",
        ]
        if not _env_bool("GOOGLE_DRIVE_RESULTS_INCLUDE_TENSORBOARD", True):
            cmd.extend(["--filter", "- *.tfevents.*"])
        if verbose:
            cmd.extend(["--progress", "--stats-one-line", "--log-level", "INFO"])
    else:
        dest_path = pathlib.Path(destination).expanduser()
        dest_path.mkdir(parents=True, exist_ok=True)
        rsync = shutil.which("rsync")
        if rsync:
            cmd = [rsync, "-a", f"{source_path}/" if source_path.is_dir() else str(source_path), str(dest_path)]
        else:
            cmd = ["cp", "-a", str(source_path), str(dest_path)]

    print(f"Google Drive sync: {source_path} -> {destination}", flush=True)
    if background and not verbose:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, check=False)
    return True
