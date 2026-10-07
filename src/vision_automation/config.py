"""Central configuration. Tunable values for the grounding search live here."""

from dataclasses import dataclass
from pathlib import Path

API_KEY_ENV = "ANTHROPIC_API_KEY"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEBUG_DIR = REPO_ROOT / "debug"
ANNOTATED_DIR = REPO_ROOT / "output" / "annotated"
TRACE_DIR = DEBUG_DIR / "traces"

PROJECT_FOLDER_NAME = "tjm-project"
POSTS_URL = "https://jsonplaceholder.typicode.com/posts"
POST_COUNT = 10


@dataclass(frozen=True)
class Config:
    # --- model ---
    model: str = "claude-sonnet-5-5"
    effort: str = "medium"  # sent as output_config.effort
    max_tokens: int = 4096
    request_timeout_s: float = 60.0
    sdk_max_retries: int = 2  # SDK-level retries for 408/409/429/5xx/connection errors

    # --- scoring / search (tuned later; the paper does not give these numbers) ---
    sigma: float = 0.3  # paper: 0.3
    dilate_factor: float = 3.0
    dilate_min_size: int = 240  # px
    dilate_max_ratio: float = 8.0
    # Candidates overlapping a better one by more than this IoU are merged away. 0.5 let three heavily
    # overlapping left-column candidates through and crowded out the correct one in a real run.
    nms_iou: float = 0.3
    # Tuned for 1080p from manual runs: a 190x150 crop grounded correctly, a 600x400 crop and the
    # full screenshot did not. The paper's 1280 is for 4K+ screens. Recursion keeps cropping until
    # the candidate's longer side is <= this, then grounds the target directly.
    direct_ground_size: int = 300  # px (screen pixels), longer side
    # If every candidate at a level is exhausted (skipped as stalled/revisited, or failed) and the region is
    # at most this large, try grounding the target directly in it (still verified) before giving up the branch.
    fallback_direct_size: int = 640  # px, longer side; 0 disables
    view_long_side: int = 800  # px: crops are upscaled so the model sees them at this longer side
    verify_crop_size: int = 300  # px (screen pixels): window around the predicted point for the verifier
    max_depth: int = 3
    max_hints_per_level: int = 6  # area/neighbor hints grounded per level (1 grounder call each)
    dilate_skip_long_side: int = 600  # boxes at least this large are used as-is, not dilated
    stall_area_ratio: float = 0.9  # a candidate >= this fraction of its parent region is not progress
    max_llm_calls_per_find: int = 40

    # --- desktop workflow (step 8): every wait polls for a condition, these are only the give-up times ---
    launch_timeout_s: float = 20.0  # double-click -> a new Notepad window appears
    dialog_timeout_s: float = 10.0  # Save As / overwrite confirmation dialogs
    # Saving does not rest on one keystroke: the File name box is verified, Enter is tried first and, if the
    # dialog is still open save_confirm_wait_s later, Alt+S (the Save button accelerator); bounded in total.
    focus_timeout_s: float = 3.0  # waiting for a window to take the keyboard focus before sending keys
    # Typing is verified against the editor's own text after every word (Notepad's autocorrect can rewrite a
    # word, e.g. "commodi" -> "commode", and typing it again would give the same result).
    chunk_retries: int = 2  # repairs per word before the whole document is cleared and typed again
    type_retries: int = 2  # clear (Ctrl+A, Delete) and re-type the whole document this many times, then stop
    chunk_settle_s: float = 0.04  # pause after a word before reading the editor back
    save_attempts: int = 3  # confirmations: Enter, then Alt+S, then Alt+S again
    save_confirm_wait_s: float = 1.5  # how long to wait for the dialog to respond to each confirmation
    field_retries: int = 2  # re-type attempts if the File name box does not hold exactly the path
    field_read_timeout_s: float = 1.5  # how long to wait for the box to show the typed path
    close_timeout_s: float = 10.0  # Notepad window disappears after closing
    desktop_timeout_s: float = 4.0  # Win+D -> the desktop is in front
    stable_timeout_s: float = 3.0  # screenshot stops changing (animations finished)
    popup_dismissals: int = 2  # pop-ups the workflow will dismiss before one find, then it stops
    park_settle_s: float = 0.4  # after parking the cursor: let hover tooltips fade (nothing to poll for)


CONFIG = Config()
