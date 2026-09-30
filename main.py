#!/usr/bin/env python3

from __future__ import annotations

import ast
import csv
import importlib.util
import logging
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Optional, Set
from urllib.parse import urljoin, urlparse

import common
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("mapanything_downloader")


def _cfg(key: str, default: Any = None) -> Any:
    """Read an optional config value and fall back to a default if missing."""
    try:
        value = common.get_configs(key)
    except KeyError:
        return default
    return default if value is None else value


cfg = SimpleNamespace(
    csv_file=common.get_configs("mapping"),  # path to mapping CSV
    videos_root=common.get_configs("videos"),  # list of configured video folder names
    base_url=common.get_configs("base_url"),  # base file server URL
    token=_cfg("token", None),  # optional token used in requests
    timeout=_cfg("timeout", 20),  # request timeout in seconds
    max_pages=_cfg("max_pages", 500),  # crawl depth limit when browsing aliases
    debug=_cfg("debug", True),  # enable verbose downloader logging
    aliases=_cfg("aliases", ["tue1", "tue2", "tue3", "tue4"]),  # server aliases to try
    download_dir=_cfg("download_dir", "downloads"),  # temporary folder for downloaded videos
    runs_dir=_cfg("RUNS_DIR", "runs"),  # folder where JSONL outputs and frame folders are written
    mapanything_fps=_cfg("MAPANYTHING_FPS", 0.01),  # frame sampling FPS used for extraction and timestamps
    enable_viz=_cfg("ENABLE_VIZ", False),  # reserved flag, currently unused in this script
    delete_downloaded_video=_cfg("DELETE_DOWNLOADED_VIDEO", True),  # delete downloaded video after processing
    delete_frames_after_processing=_cfg("DELETE_FRAMES_AFTER_PROCESSING", True),  # delete frames after inference
    keep_going=_cfg("KEEP_GOING", True),  # continue with the next video if one fails
    overwrite_frames=_cfg("OVERWRITE_FRAMES", True),  # re extract frames even if frame files already exist
    skip_existing_outputs=_cfg("SKIP_EXISTING_OUTPUTS", True),  # skip segments whose JSONL output already exists
    target_locality=_cfg("TARGET_LOCALITY", None),  # optional CSV locality filter
    target_row_id=_cfg("TARGET_ROW_ID", None),  # optional CSV row id filter
    max_videos_to_process=_cfg("MAX_VIDEOS_TO_PROCESS", None),  # optional cap on queued source videos
)

secrets = SimpleNamespace(
    ftp_username=common.get_secrets("ftp_username"),
    ftp_password=common.get_secrets("ftp_password"),
)


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def sanitise_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^\w\s.-]", "", value)
    value = re.sub(r"\s+", "_", value)
    return value or "unknown"


def format_time_label(value: float | int | str) -> str:
    try:
        num = float(value)
        if num.is_integer():
            return str(int(num))
        return str(num).replace(".", "p")
    except Exception:
        return sanitise_name(str(value))


def segment_output_path(runs_dir: Path, video_id: str, start_time: float) -> Path:
    return runs_dir / f"{sanitise_name(video_id)}_{format_time_label(start_time)}.jsonl"


def parse_videos_field(value: str) -> list[str]:
    """
    Converts:
    "[abc,def,ghi]"
    into:
    ["abc", "def", "ghi"]
    """
    if value is None:
        return []

    text = value.strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    if not text.strip():
        return []

    parts = [x.strip().strip('"').strip("'") for x in text.split(",")]
    return [x for x in parts if x]


def parse_nested_times_field(value: str) -> list[list[float]]:
    """
    Converts strings like:
    "[[50],[51],[29,821]]"
    into:
    [[50.0], [51.0], [29.0, 821.0]]
    """
    if value is None:
        return []

    text = value.strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        logger.warning(f"Could not parse time field: {text}")
        return []

    result: list[list[float]] = []

    if not isinstance(parsed, list):
        return result

    for item in parsed:
        if isinstance(item, list):
            cleaned: list[float] = []
            for x in item:
                try:
                    cleaned.append(float(x))
                except Exception:
                    continue
            result.append(cleaned)
        else:
            try:
                result.append([float(item)])
            except Exception:
                result.append([])

    return result


def get_video_info_ffprobe(video_path: str) -> tuple[str, float]:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return "unknown", 0.0

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
        if len(lines) < 3:
            return "unknown", 0.0

        width = int(lines[0])
        height = int(lines[1])
        avg_frame_rate = lines[2]

        fps = 0.0
        if "/" in avg_frame_rate:
            num, den = avg_frame_rate.split("/", 1)
            den_value = float(den)
            if den_value != 0:
                fps = float(num) / den_value
        else:
            fps = float(avg_frame_rate)

        resolution = f"{height}p" if height > 0 else f"{width}x{height}"
        return resolution, fps
    except Exception as e:
        logger.warning(f"ffprobe metadata extraction failed for {video_path}: {e}")
        return "unknown", 0.0


_local_video_index: dict[Path, dict[str, Path]] = {}


def _get_local_video_index(root_path: Path) -> dict[str, Path]:
    """Map file names to paths for all files under root_path. Built once per root."""
    if root_path not in _local_video_index:
        index: dict[str, Path] = {}
        for path in sorted(root_path.rglob("*")):
            if path.is_file():
                index.setdefault(path.name, path)
        _local_video_index[root_path] = index
        logger.info(f"Indexed {len(index)} files in local video folder: {root_path}")
    return _local_video_index[root_path]


def find_existing_local_video(
    filename: str,
    repo_root: Path,
    video_roots: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> Optional[Path]:
    if not video_roots:
        return None

    filename_with_ext = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
    candidate_names = [filename_with_ext]
    if filename not in candidate_names:
        candidate_names.append(filename)

    roots = video_roots if isinstance(video_roots, (list, tuple)) else [video_roots]

    for root_value in roots:
        root_path = resolve_path(root_value, repo_root)
        if not root_path.exists():
            logger.debug(f"Configured local video folder does not exist: {root_path}")
            continue

        for candidate_name in candidate_names:
            direct_path = root_path / candidate_name
            if direct_path.exists() and direct_path.is_file():
                logger.info(f"Using existing local video: {direct_path}")
                return direct_path.resolve()

        index = _get_local_video_index(root_path)
        for candidate_name in candidate_names:
            match = index.get(candidate_name)
            if match is not None:
                logger.info(f"Using existing local video: {match}")
                return match.resolve()

    return None


def extract_frames_for_segment(
    ffmpeg_bin: str,
    video_path: Path,
    frames_dir: Path,
    fps: float,
    start_time: float,
    end_time: float,
    overwrite_frames: bool,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = list(frames_dir.glob("frame_*.jpg"))
    if existing_frames and not overwrite_frames:
        logger.info(f"Frames already exist for {video_path.name}, skipping extraction")
        return

    if overwrite_frames:
        for old_file in frames_dir.glob("*"):
            if old_file.is_file():
                old_file.unlink()

    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(
            f"Invalid segment for {video_path.name}: start={start_time}, end={end_time}"
        )

    output_pattern = frames_dir / "frame_%05d.jpg"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        str(start_time),
        "-i",
        str(video_path),
        "-t",
        str(duration),
        "-vf",
        f"fps={fps}",
        str(output_pattern),
    ]
    run_command(cmd)

    extracted = list(frames_dir.glob("frame_*.jpg"))
    if not extracted:
        raise RuntimeError(
            f"No frames extracted for {video_path.name} from {start_time} to {end_time}"
        )


class PoseExporter:
    """Loads the MapAnything model on first use and keeps it for all segments."""

    def __init__(self) -> None:
        self._model: Any = None
        self._export_module: Any = None  # scripts/demo_images_pose_export.py, loaded on first use

    def export(
        self,
        frames_dir: Path,
        output_jsonl: Path,
        video_id: str,
        segment_start_time: float,
    ) -> None:
        if self._export_module is None:
            # Loaded lazily so torch and the model are only loaded when there is work to do.
            # Loaded by path because an installed package also provides a top level "scripts" module.
            script_path = Path(__file__).resolve().parent / "scripts" / "demo_images_pose_export.py"
            spec = importlib.util.spec_from_file_location("demo_images_pose_export", script_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load {script_path}")
            # Any: a module loaded by path has no static type, and older typeshed types the loader too loosely
            module: Any = importlib.util.module_from_spec(spec)
            loader: Any = spec.loader
            loader.exec_module(module)

            self._model = module.load_model()
            self._export_module = module

        self._export_module.export_poses(
            model=self._model,
            image_folder=frames_dir,
            output_path=output_jsonl,
            video_id=video_id,
            segment_start_time=segment_start_time,
            fps=float(cfg.mapanything_fps),
        )


def load_video_jobs_from_mapping(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"mapping csv not found: {csv_path}")

    jobs: list[dict] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row_id = (row.get("id") or "").strip()
            locality = (row.get("locality") or "").strip()

            if cfg.target_row_id is not None and row_id != str(cfg.target_row_id):
                continue

            if cfg.target_locality is not None and locality != cfg.target_locality:
                continue

            videos = parse_videos_field(row.get("videos") or "")
            start_lists = parse_nested_times_field(row.get("start_time") or "")
            end_lists = parse_nested_times_field(row.get("end_time") or "")

            count = min(len(videos), len(start_lists), len(end_lists))
            if count == 0:
                continue

            if len(videos) != len(start_lists) or len(videos) != len(end_lists):
                logger.warning(
                    f"Length mismatch in row id={row_id} locality={locality}. "
                    f"videos={len(videos)} starts={len(start_lists)} ends={len(end_lists)}. "
                    f"Using first {count} items."
                )

            for idx in range(count):
                video_id = videos[idx]
                starts = start_lists[idx]
                ends = end_lists[idx]

                seg_count = min(len(starts), len(ends))
                if seg_count == 0:
                    logger.warning(
                        f"No valid segments for video {video_id} in row id={row_id}"
                    )
                    continue

                segments: list[tuple[float, float]] = []
                for seg_idx in range(seg_count):
                    start_value = starts[seg_idx]
                    end_value = ends[seg_idx]

                    if end_value <= start_value:
                        logger.warning(
                            f"Skipping invalid segment for video {video_id}: "
                            f"start={start_value}, end={end_value}"
                        )
                        continue

                    segments.append((start_value, end_value))

                if not segments:
                    continue

                jobs.append(
                    {
                        "row_id": row_id,
                        "locality": locality,
                        "video_id": video_id,
                        "segments": segments,
                    }
                )

                if (
                    cfg.max_videos_to_process is not None
                    and len(jobs) >= cfg.max_videos_to_process
                ):
                    return jobs

    return jobs


def download_video_from_server(
    filename: str,
    base_url: Optional[str],
    out_dir: str | Path = ".",
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 20,
    debug: bool = True,
    max_pages: int = 500,
    aliases: Optional[list[str]] = None,
) -> Optional[tuple[str, str, str, float]]:
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if not base_url:
        logger.error("Base URL is missing.")
        return None

    base = base_url if base_url.endswith("/") else base_url + "/"

    if username == "":
        username = None
    if password == "":
        password = None

    aliases = aliases or ["tue1", "tue2", "tue3", "tue4"]

    filename_with_ext = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
    filename_lower = filename_with_ext.lower()

    out_dir_path = Path(out_dir)
    req_params = {"token": token} if token else None

    logger.info(f"Starting download for '{filename_with_ext}'")
    logger.debug(
        f"Base URL: {base} | Auth: {'Basic' if username and password else 'None'} | Token: {'Yes' if token else 'No'}"
    )

    with requests.Session() as session:
        if username and password:
            session.auth = (username, password)
        session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

        def fetch(url: str, stream: bool = False) -> Optional[requests.Response]:
            try:
                response = session.get(
                    url,
                    timeout=timeout,
                    params=req_params,
                    stream=stream,
                )
                logger.debug(f"GET {url} -> {response.status_code}")
                if response.status_code == 401:
                    logger.error(f"Authentication failed for {url}")
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Request failed [{url}]: {e}")
                return None

        def save_response_to_file(response: requests.Response, local_path: Path) -> bool:
            try:
                total = int(response.headers.get("content-length", 0)) or None
                written = 0

                local_path.parent.mkdir(parents=True, exist_ok=True)
                if local_path.exists():
                    local_path.unlink()

                with open(local_path, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading: {local_path.name}",
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                            if total:
                                bar.update(len(chunk))

                logger.info(f"Download complete: {local_path} ({written} bytes)")
                return True
            except Exception as e:
                logger.error(f"Download failed for {local_path.name}: {e}")
                return False

        def download(response: requests.Response) -> Optional[tuple[str, str, str, float]]:
            """Save the response as the video file and return (path, video id, resolution, fps)."""
            local_path = out_dir_path / filename_with_ext
            if not save_response_to_file(response, local_path):
                return None
            resolution, fps = get_video_info_ffprobe(str(local_path))
            logger.info(f"Saved '{filename_with_ext}' (res={resolution}, fps={fps})")
            return str(local_path), Path(filename_with_ext).stem, resolution, fps

        for alias in aliases:
            direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
            logger.debug(f"Trying direct URL: {direct_url}")

            response = fetch(direct_url, stream=True)
            if response is None:
                continue

            logger.info(f"Found file via direct URL: {direct_url}")
            return download(response)

        visited: Set[str] = set()

        def is_dir_link(href: str) -> bool:
            return href.startswith("/v/") and "/browse" in href

        def is_file_link(href: str) -> bool:
            return "/files/" in href

        def crawl(start_url: str) -> Optional[str]:
            stack = [start_url]
            pages_seen = 0

            while stack:
                url = stack.pop()

                if url in visited:
                    continue

                visited.add(url)
                pages_seen += 1
                if pages_seen > max_pages:
                    logger.warning(f"Crawl aborted after {max_pages} pages.")
                    return None

                response = fetch(url)
                if response is None:
                    continue

                try:
                    soup = BeautifulSoup(response.text, "html.parser")
                except Exception as e:
                    logger.warning(f"HTML parse failed at {url}: {e}")
                    continue

                for a in soup.find_all("a"):
                    href = (a.get("href") or "").strip()  # type: ignore
                    if not href:
                        continue

                    full = urljoin(url, href)

                    if is_file_link(href):
                        anchor_text = (a.text or "").strip().lower()
                        tail = PurePosixPath(urlparse(full).path).name.lower()
                        if anchor_text == filename_lower or tail == filename_lower:
                            logger.info(f"File located via crawl: {full}")
                            return full

                    if is_dir_link(href):
                        stack.append(full)

            logger.debug("Crawl finished, no file found.")
            return None

        for alias in aliases:
            start_url = urljoin(base, f"v/{alias}/browse")
            logger.debug(f"Crawling alias: {alias} -> {start_url}")

            found = crawl(start_url)
            if not found:
                continue

            response = fetch(found, stream=True)
            if response is None:
                continue

            return download(response)

        logger.warning(f"File '{filename_with_ext}' not found in any alias.")
        return None


def process_segments_for_video(
    runs_dir: Path,
    ffmpeg_bin: str,
    exporter: PoseExporter,
    downloaded_video_path: Path,
    video_id: str,
    segments: list[tuple[float, float]],
) -> list[Path]:
    outputs: list[Path] = []
    video_work_dir = runs_dir / sanitise_name(video_id)
    video_work_dir.mkdir(parents=True, exist_ok=True)

    for idx, (start_time, end_time) in enumerate(segments, start=1):
        frames_dir = video_work_dir / f"frames_{format_time_label(start_time)}"
        output_jsonl = segment_output_path(runs_dir, video_id, start_time)

        logger.info(
            f"Processing segment {idx}/{len(segments)} for {video_id}: "
            f"start={start_time}, end={end_time}"
        )

        extract_frames_for_segment(
            ffmpeg_bin=ffmpeg_bin,
            video_path=downloaded_video_path,
            frames_dir=frames_dir,
            fps=float(cfg.mapanything_fps),
            start_time=start_time,
            end_time=end_time,
            overwrite_frames=bool(cfg.overwrite_frames),
        )

        exporter.export(
            frames_dir=frames_dir,
            output_jsonl=output_jsonl,
            video_id=video_id,
            segment_start_time=start_time,
        )

        outputs.append(output_jsonl)
        logger.info(f"Saved JSONL: {output_jsonl}")

        if bool(cfg.delete_frames_after_processing) and frames_dir.exists():
            shutil.rmtree(frames_dir)
            logger.info(f"Deleted frames: {frames_dir}")

    if bool(cfg.delete_frames_after_processing) and video_work_dir.exists() and not any(video_work_dir.iterdir()):
        video_work_dir.rmdir()

    return outputs


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    mapping_csv_path = resolve_path(cfg.csv_file, repo_root)
    downloads_dir = resolve_path(cfg.download_dir, repo_root)
    runs_dir = resolve_path(cfg.runs_dir, repo_root)

    downloads_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        logger.error("ffmpeg was not found on your PATH")
        logger.error("Install it first, for example: brew install ffmpeg")
        return 1

    demo_script = repo_root / "scripts" / "demo_images_pose_export.py"
    if not demo_script.exists():
        logger.error(f"Could not find: {demo_script}")
        logger.error("Create scripts/demo_images_pose_export.py in the root of the map anything repo.")
        return 1

    exporter = PoseExporter()
    skipped_segments = 0

    logger.info(f"Mapping CSV: {mapping_csv_path}")
    logger.info(f"Temporary download directory: {downloads_dir}")
    logger.info(f"Runs directory: {runs_dir}")
    logger.info(f"Configured video folders: {cfg.videos_root}")

    try:
        jobs = load_video_jobs_from_mapping(mapping_csv_path)
    except Exception as exc:
        logger.error(str(exc))
        return 1

    if not jobs:
        logger.error("No video jobs found in mapping csv with current filters.")
        return 1

    logger.info(f"Loaded {len(jobs)} source videos from {mapping_csv_path}")

    failures: list[tuple[str, str]] = []

    for index, job in enumerate(jobs, start=1):
        row_id = job["row_id"]
        locality = job["locality"]
        video_id = job["video_id"]
        segments = job["segments"]

        if bool(cfg.skip_existing_outputs):
            pending = [
                seg for seg in segments
                if not segment_output_path(runs_dir, video_id, seg[0]).exists()
            ]
            skipped_segments += len(segments) - len(pending)
            if not pending:
                logger.info(
                    f"[{index}/{len(jobs)}] Skipping {video_id}: all {len(segments)} segments already exported"
                )
                continue
            if len(pending) < len(segments):
                logger.info(f"{video_id}: {len(segments) - len(pending)} segments already exported, skipping them")
            segments = pending

        logger.info("=" * 80)
        logger.info(
            f"[{index}/{len(jobs)}] Starting video_id={video_id} locality={locality} row_id={row_id}"
        )
        logger.info(f"Segments to process: {len(segments)}")
        logger.info("=" * 80)

        video_path_to_process: Optional[Path] = None
        downloaded_from_server = False

        try:
            existing_local_video = find_existing_local_video(
                filename=video_id,
                repo_root=repo_root,
                video_roots=cfg.videos_root,
            )

            if existing_local_video is not None:
                video_path_to_process = existing_local_video
                resolution, fps = get_video_info_ffprobe(str(video_path_to_process))
                logger.info(
                    f"Using local video {video_path_to_process.name} | resolution={resolution} | fps={fps}"
                )
            else:
                result = download_video_from_server(
                    filename=video_id,
                    base_url=cfg.base_url,
                    out_dir=downloads_dir,
                    username=secrets.ftp_username,
                    password=secrets.ftp_password,
                    token=cfg.token,
                    timeout=int(cfg.timeout),
                    debug=bool(cfg.debug),
                    max_pages=int(cfg.max_pages),
                    aliases=list(cfg.aliases),
                )

                if result is None:
                    raise RuntimeError(f"Download failed or file not found for: {video_id}")

                local_path, _, resolution, fps = result
                video_path_to_process = Path(local_path)
                downloaded_from_server = True

                logger.info(
                    f"Downloaded {video_path_to_process.name} | resolution={resolution} | fps={fps}"
                )

            output_files = process_segments_for_video(
                runs_dir=runs_dir,
                ffmpeg_bin=ffmpeg_bin,
                exporter=exporter,
                downloaded_video_path=video_path_to_process,
                video_id=video_id,
                segments=segments,
            )

            logger.info(f"Finished pose export for {video_id}")
            for out_file in output_files:
                logger.info(f"Output JSONL: {out_file}")

        except Exception as exc:
            failures.append((video_id, str(exc)))
            logger.error(f"Failed: {video_id}")
            logger.error(str(exc))

            if not bool(cfg.keep_going):
                break

        finally:
            if (
                downloaded_from_server
                and video_path_to_process
                and video_path_to_process.exists()
                and bool(cfg.delete_downloaded_video)
            ):
                video_path_to_process.unlink()
                logger.info(f"Deleted downloaded video: {video_path_to_process}")

    logger.info("=" * 80)
    logger.info("Finished all queued videos")
    if skipped_segments:
        logger.info(f"Skipped {skipped_segments} segments with existing outputs")
    logger.info("=" * 80)

    if failures:
        logger.error("Some videos failed:")
        for video_id, error in failures:
            logger.error(f"{video_id}: {error}")
        return 1

    logger.info("All videos processed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
