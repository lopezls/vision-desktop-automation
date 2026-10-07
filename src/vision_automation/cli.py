"""Command line entry point. `check` verifies setup; later steps add `run`, `locate`, `plan`."""

import argparse
import logging
import sys
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw

from .config import CONFIG
from .llm import LLM, LLMError, MissingApiKeyError, make_client


def _check() -> int:
    ok = True
    client = make_client()
    print(f"API key: present, model under test: {CONFIG.model}, effort: {CONFIG.effort}")

    # 1. Model ID exists in the API model list.
    ids = [m.id for m in client.models.list()]
    if CONFIG.model in ids:
        print(f"[ok]   model '{CONFIG.model}' is in the API model list ({len(ids)} models)")
    else:
        ok = False
        print(f"[FAIL] model '{CONFIG.model}' not in model list. Available: {', '.join(ids)}")

    # 2. Effort is actually sent AND read by the server: a bogus value must be rejected.
    llm = LLM(client)
    print(f"[info] request params sent: {llm.request_params(messages='...')['output_config']}")
    try:
        client.messages.create(
            **llm.request_params(
                max_tokens=16,
                output_config={"effort": "definitely-not-a-level"},
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        ok = False
        print("[FAIL] bogus effort value was accepted -> server may be ignoring output_config.effort")
    except anthropic.BadRequestError as e:
        print(f"[ok]   bogus effort rejected with 400 -> server reads output_config.effort ({e.message[:90]})")

    # 3. One real vision call on a generated image.
    img = Image.new("RGB", (400, 200), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 60, 160, 140], fill="red")
    d.ellipse([240, 60, 360, 140], fill="blue")
    reply = llm.ask("check", "What colors are the two shapes, left to right? Answer in one short line.", [img])
    print(f"[ok]   vision call reply: {reply.strip()!r}")
    return 0 if ok else 1


def _screenshot(delay: float, mark: str | None) -> int:
    import time

    from .annotate import annotate_result, save_image
    from .llm import timed
    from .screen import capture, screen_size
    from .types import Point

    print(f"Screen size reported: {screen_size()}. Capturing in {delay:g}s...")
    time.sleep(delay)
    with timed("capture"):
        img = capture()
    print(f"Captured {img.width}x{img.height}")
    path = save_image(img, "screenshot")
    print(f"Saved {path}")
    if mark:
        x, y = (float(v) for v in mark.split(","))
        marked = annotate_result(img, Point(x, y), label="marker test")
        print(f"Saved {save_image(marked, 'screenshot_marked')}")
    return 0


def _parse_box(text: str):
    from .types import BBox

    try:
        x0, y0, x1, y1 = (float(v) for v in text.split(","))
    except ValueError:
        raise SystemExit(f"--crop must be 'x0,y0,x1,y1' in pixels, got {text!r}")
    return BBox(x0, y0, x1, y1)


def _load_view(args, input_name: str):
    """Load/capture the image and apply --crop/--zoom. Returns (full image, Crop|None, image the model sees)."""
    import time

    from .annotate import save_image
    from .screen import capture, crop_region

    if args.image:
        img = Image.open(args.image).convert("RGB")
        print(f"Loaded {args.image} ({img.width}x{img.height})")
    else:
        print(f"Capturing screen in {args.delay:g}s (show the desktop now)...")
        time.sleep(args.delay)
        img = capture()
        print(f"Captured {img.width}x{img.height}")
        print(f"Saved {save_image(img, input_name)}")

    crop = None
    view = img
    if args.crop:
        crop = crop_region(img, _parse_box(args.crop), target_long_side=args.zoom)
        view = crop.image
        print(f"Crop region (full-image px): {crop.box.as_ints()}  zoom x{crop.scale:.2f}  "
              f"-> model sees {view.width}x{view.height}")
    elif args.zoom:
        print("--zoom only applies together with --crop; ignoring")
    return img, crop, view


def _plan(args) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from .annotate import RED, YELLOW, draw_box, save_image
    from .grounder import ClaudeGrounder
    from .planner import ClaudePlanner

    img, crop, view = _load_view(args, "plan_input")
    llm = LLM()
    planner = ClaudePlanner(llm)

    if not args.no_popup:
        pop = planner.check_popup(img, args.query)
        print(f"POPUP: blocked={pop.blocked} action={pop.action} label={pop.control_label!r} "
              f"popup={pop.popup!r}")
        print(f"       reason: {pop.reason}")

    inf = planner.infer_positions(view, args.query, is_crop=crop is not None)
    if inf.no_target:
        print("PLAN: 'No target' / no usable hints at this level (a failed level)")
        return 1
    print("PLAN hints (best first):")
    for h in inf.hints:
        print(f"  {h.rank}. <{h.kind}> {h.text}")

    if args.ground:
        grounder = ClaudeGrounder(llm)
        hints = inf.of_kind("area", "neighbor")
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda h: grounder.ground(view, h.text), hints))
        out = draw_box(img, crop.box, YELLOW, 2, label="crop") if crop else img
        for h, r in zip(hints, results):
            if not r.found:
                print(f"  grounded <{h.kind}> {h.text!r}: not found ({r.note})")
                continue
            box = crop.box_to_parent(r.box) if crop else r.box
            c = box.center
            print(f"  grounded <{h.kind}> {h.text!r}: box={box.as_ints()} center=({c.x:.0f}, {c.y:.0f})")
            out = draw_box(out, box, RED if h.kind == "area" else (0, 160, 255), 2, label=f"{h.kind[0]}{h.rank}")
        print(f"Saved {save_image(out, 'plan_overlay')}")
    return 0


def _verify(args) -> int:
    from .annotate import save_image
    from .verifier import ClaudeVerifier

    img = Image.open(args.image).convert("RGB") if args.image else None
    if img is None:
        import time

        from .screen import capture

        print(f"Capturing screen in {args.delay:g}s...")
        time.sleep(args.delay)
        img = capture()
    x, y = (float(v) for v in args.point.split(","))
    box = _parse_box(args.box) if args.box else None
    from .types import Point

    verdict = ClaudeVerifier(LLM()).verify(img, Point(x, y), args.query, box)
    print(f"VERDICT: {verdict.result}" + (f"  new_instruction={verdict.new_instruction!r}" if verdict.new_instruction else ""))
    if verdict.reason:
        print(f"         {verdict.reason}")
    print(f"Saved the crop the model saw: {save_image(verdict.marked, 'verify_crop')}")
    return 0 if verdict.is_target else 1


def _locate(args) -> int:
    """Dry run: find the target and save an annotated screenshot. Never clicks or types."""
    import time
    from dataclasses import replace

    from .annotate import annotate_result, save_image
    from .config import ANNOTATED_DIR
    from .grounder import ClaudeGrounder
    from .llm import stage_times, timed
    from .notepad import TARGET_DESCRIPTION
    from .planner import ClaudePlanner
    from .screen import capture
    from .search import GroundingError, PopupBlocked, Searcher
    from .verifier import ClaudeVerifier

    overrides = {k: v for k, v in {
        "direct_ground_size": args.direct_size, "max_depth": args.max_depth,
        "max_llm_calls_per_find": args.max_calls}.items() if v is not None}
    cfg = replace(CONFIG, **overrides)
    description = args.description or TARGET_DESCRIPTION

    if args.image:
        img = Image.open(args.image).convert("RGB")
        print(f"Loaded {args.image} ({img.width}x{img.height})")
    else:
        print(f"Capturing screen in {args.delay:g}s (show the desktop now)...")
        time.sleep(args.delay)
        with timed("capture"):
            img = capture()
        print(f"Captured {img.width}x{img.height}")

    llm = LLM(cfg=cfg)
    searcher = Searcher(ClaudePlanner(llm), ClaudeGrounder(llm), ClaudeVerifier(llm, cfg), llm, cfg)
    print(f"Target: {description!r}  (direct size {cfg.direct_ground_size}px, max depth {cfg.max_depth}, "
          f"call cap {cfg.max_llm_calls_per_find})")

    def report(trace_path, calls) -> None:
        print(f"Search calls: {calls}  unusable model replies: {len(llm.parse_failures)}")
        for stage, t in stage_times().items():
            print(f"  {stage:<14} {t['seconds']:6.1f}s over {t['calls']} call(s)")
        print(f"Trace: {trace_path}")

    try:
        located = searcher.find(img, description, check_popup=not args.no_popup)
    except PopupBlocked as e:
        print(f"POP-UP BLOCKING THE SCREEN (not clicked): {e}")
        report(e.trace_path, e.calls)
        return 3
    except GroundingError as e:
        print(f"NOT FOUND: {e}")
        if e.image_path:
            print(f"Debug image: {e.image_path}")
        report(e.trace_path, e.calls)
        return 1

    c = located.point
    print(f"FOUND  click point=({c.x:.0f}, {c.y:.0f})  box={located.box.as_ints()}  "
          f"depth={located.depth}  verdict={located.verdict.result}  {located.elapsed_s:.1f}s")
    out = annotate_result(img, c, located.box, label=args.label)
    out_dir = Path(args.out_dir) if args.out_dir else ANNOTATED_DIR
    if args.name:
        path = save_image(out, args.name, out_dir, timestamp=False)
    else:
        path = save_image(out, "locate", out_dir)
    print(f"Annotated image: {path}")
    report(located.trace_path, located.calls)
    return 0


def _ground(args) -> int:
    from .annotate import YELLOW, annotate_result, draw_box, save_image
    from .grounder import ClaudeGrounder
    from .types import Point

    img, crop, view = _load_view(args, 'ground_input')

    grounder = ClaudeGrounder(LLM())
    res = grounder.ground(view, args.query)
    if not res.found:
        print(f"NOT FOUND: {res.note}")
        return 1

    # Map back to full-image pixels so full-screen and cropped runs are directly comparable.
    box = crop.box_to_parent(res.box) if crop else res.box
    center = box.center
    print(f"FOUND  box(full-image px)={box.as_ints()}  center=({center.x:.0f}, {center.y:.0f})  "
          f"confidence={res.confidence}  note={res.note!r}")
    if crop:
        local = res.box
        print(f"       box(crop px)={local.as_ints()}  center=({local.center.x:.0f}, {local.center.y:.0f})")

    out = img
    if crop:
        out = draw_box(out, crop.box, YELLOW, 2, label="crop")
    out = annotate_result(out, Point(center.x, center.y), box, label="ground")
    print(f"Saved {save_image(out, 'ground_crop' if crop else 'ground_full')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vision-automation")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="verify API key, model id, effort param and a vision call")
    gr = sub.add_parser("ground", help="ground a text query on a screenshot (optionally in a crop)")
    gr.add_argument("query", help="description of the element to find")
    gr.add_argument("--image", help="path to a saved screenshot (default: capture the screen)")
    gr.add_argument("--delay", type=float, default=3.0, help="seconds before capturing when no --image")
    gr.add_argument("--crop", help="x0,y0,x1,y1 in full-image pixels; ground only inside this region")
    gr.add_argument("--zoom", type=int, help="with --crop: resize the crop's longer side to N px")
    pl = sub.add_parser("plan", help="popup check + position-inference hints for a query")
    pl.add_argument("query", help="description of the target")
    pl.add_argument("--image", help="path to a saved screenshot (default: capture the screen)")
    pl.add_argument("--delay", type=float, default=3.0)
    pl.add_argument("--crop", help="x0,y0,x1,y1 in full-image pixels; plan only inside this region")
    pl.add_argument("--zoom", type=int, help="with --crop: resize the crop's longer side to N px")
    pl.add_argument("--no-popup", action="store_true", help="skip the popup check")
    pl.add_argument("--ground", action="store_true", help="also ground each area/neighbor hint and save an overlay")
    lc = sub.add_parser("locate", help="find the target icon and save an annotated screenshot (never clicks)")
    lc.add_argument("--description", help="target description (default: the Notepad desktop icon)")
    lc.add_argument("--image", help="path to a saved screenshot (default: capture the screen)")
    lc.add_argument("--delay", type=float, default=3.0, help="seconds before capturing when no --image")
    lc.add_argument("--name", help="output file name stem in output/annotated/ (e.g. top-left); overwrites")
    lc.add_argument("--label", default="Notepad icon", help="label drawn on the annotated image")
    lc.add_argument("--out-dir", help="where to save the annotated image (default: output/annotated/)")
    lc.add_argument("--no-popup", action="store_true", help="skip the popup check")
    lc.add_argument("--direct-size", type=int, help="retune direct_ground_size (px) for this run")
    lc.add_argument("--max-depth", type=int, help="override max recursion depth")
    lc.add_argument("--max-calls", type=int, help="override the per-find LLM call cap")
    vf = sub.add_parser("verify", help="verify that a point on a screenshot is the target")
    vf.add_argument("query", help="description of the target")
    vf.add_argument("--point", required=True, help="x,y in full-image pixels")
    vf.add_argument("--box", help="optional x0,y0,x1,y1 (full-image px) to draw as the red box")
    vf.add_argument("--image", help="path to a saved screenshot (default: capture the screen)")
    vf.add_argument("--delay", type=float, default=3.0)
    shot = sub.add_parser("screenshot", help="capture the screen to debug/; optionally mark a point")
    shot.add_argument("--delay", type=float, default=3.0, help="seconds to wait before capturing")
    shot.add_argument("--mark", help="x,y pixel to annotate with a marker (e.g. 960,540)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        if args.cmd == "check":
            return _check()
        if args.cmd == "locate":
            return _locate(args)
        if args.cmd == "verify":
            return _verify(args)
        if args.cmd == "plan":
            return _plan(args)
        if args.cmd == "ground":
            return _ground(args)
        if args.cmd == "screenshot":
            return _screenshot(args.delay, args.mark)
    except (MissingApiKeyError, LLMError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
