"""
generate_demo_gif.py

Renders the README's demo GIF (`assets/demo.gif`) -- a terminal session showing
the *live* Cloud Run service scoring two real transactions.

Why this script is committed alongside the GIF it produces: a binary asset in a
repository whose whole method is reviewable diffs and checksummed artifacts is
an opaque blob nobody can verify. The script makes it reproducible -- run it and
the GIF is rebuilt from whatever the deployed service actually returns today.
That is also why it calls the API rather than replaying a saved transcript: a
recorded transcript can drift from the service while still looking correct, and
this repo has been caught by exactly that class of stale evidence before
(AUDIT.md section 5, "documentation asserting evidence that does not exist").

The two transactions are real rows from the committed
`dashboard/sample_transactions.csv`, not invented payloads:

  * step 60, TRANSFER, 581,421.85, origin drained to zero -- isFraud = 1
  * step 60, PAYMENT, 4,821.33, to a merchant destination -- isFraud = 0

Both are sent exactly as the API's request schema accepts them (no isFraud
field -- the service never sees a label).

    python tasks.py demo-gif
    python src/generate_demo_gif.py --api-base http://127.0.0.1:8000   # local

Requires Pillow, pinned in `requirements-dev.txt`: this is a maintenance tool,
not part of training or serving. An earlier version of this docstring claimed
Pillow was "already a training-environment dependency (matplotlib pulls it in)"
-- both halves were false. matplotlib is in none of the three requirements
files, and Pillow is present in a full dev environment only because Streamlit
happens to require it. Relying on that would have broken the documented path
(`pip install -r requirements-train.txt` then `python tasks.py demo-gif`) for
anyone who did not also install the dashboard's dependencies.

Nothing in the serving image imports this module.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import PROJECT_ROOT

DEFAULT_API = "https://fraud-api-amj2cl4jhq-uc.a.run.app"
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "demo.gif"

# GitHub-dark-ish palette, chosen so the GIF reads on both README themes.
BG = (13, 17, 23)
FG = (201, 209, 217)
DIM = (110, 118, 129)
PROMPT = (126, 231, 135)
STRING = (165, 214, 255)
ALERT = (255, 123, 114)
OK = (86, 211, 100)
ACCENT = (210, 168, 255)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\lucon.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
FONT_SIZE = 15
PAD = 16
LINE_GAP = 5
COLS = 74

# Frame budget. A README hero image is downloaded on every page view, so size is
# a feature: keep the result well under a megabyte. The levers, in order of
# effect: `disposal` (2 forces a full-frame rewrite and defeats delta encoding),
# frame count, then canvas width.
#
# Total runtime matters as much as bytes. PIL's optimizer coalesces identical
# held frames and sums their durations, so the written frame count is lower
# than the count built here while the wall-clock loop length is unchanged --
# the first version played for 55 s, which on a README reads as a static
# screenshot. Target is ~10-15 s per loop; verify against the WRITTEN file,
# not against len(frames).
#
# Holds are the right lever for dwell time, not FRAME_MS: identical held frames
# coalesce into ONE written frame carrying the summed duration, so a long hold
# costs readable seconds at almost no bytes, while FRAME_MS governs how fast the
# typing reads. Cutting both at once took an early version to 3.8 s -- too fast
# to read the response, which is the whole point of the demo.
FRAME_MS = 60
TYPE_CHARS_PER_FRAME = 5
HOLD_SHORT = 18
HOLD_LONG = 45
SIZE_WARN_KB = 900


def _load_font() -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, FONT_SIZE)
    raise SystemExit("No monospace TTF found. Tried:\n  " + "\n  ".join(FONT_CANDIDATES))


def post_json(url: str, payload: dict, timeout: float = 45.0) -> tuple[dict, float]:
    """POST and return (parsed body, wall-clock seconds).

    Timing is measured here rather than taken from the service's own
    `latency_ms` so the GIF can show round-trip time -- what a caller actually
    waits for -- alongside the in-process number the API reports.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, time.perf_counter() - start


def get_json(url: str, timeout: float = 45.0) -> tuple[dict, float]:
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, time.perf_counter() - start


FRAUD_TXN = {
    "step": 60, "type": "TRANSFER", "amount": 581421.85,
    "nameOrig": "C1021314422", "oldbalanceOrg": 581421.85, "newbalanceOrig": 0.0,
    "nameDest": "C2062695733", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}
LEGIT_TXN = {
    "step": 60, "type": "PAYMENT", "amount": 4821.33,
    "nameOrig": "C1899322999", "oldbalanceOrg": 52310.0, "newbalanceOrig": 47488.67,
    "nameDest": "M1979787155", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}


def capture(api_base: str) -> dict:
    """Hit the live service and return everything the GIF will show."""
    api_base = api_base.rstrip("/")
    print(f"capturing against {api_base}")

    # Warm the instance first, then measure. A cold start is honest but it is a
    # property of scale-to-zero rather than of the scoring path, and the README
    # states the cold-start number separately in prose.
    try:
        get_json(f"{api_base}/health")
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {api_base}: {exc}") from exc

    info, _ = get_json(f"{api_base}/model-info")
    fraud, t_fraud = post_json(f"{api_base}/score", FRAUD_TXN)
    legit, t_legit = post_json(f"{api_base}/score", LEGIT_TXN)

    for label, body in (("fraud", fraud), ("legit", legit)):
        print(f"  {label}: decision={body['decision']} "
              f"p={body['probability']!r} state_hit={body['state_hit']} "
              f"latency_ms={body['latency_ms']:.2f}")
    return {
        "api_base": api_base, "info": info,
        "fraud": fraud, "t_fraud": t_fraud,
        "legit": legit, "t_legit": t_legit,
    }


def build_script(cap: dict) -> list[tuple]:
    """The session as a list of ('type'|'out'|'hold', payload) steps.

    Every number below comes from `cap`, i.e. from the service. Nothing here is
    a literal copied from a previous run.
    """
    info, fraud, legit = cap["info"], cap["fraud"], cap["legit"]
    steps: list[tuple] = []

    steps.append(("out", [("# what is actually deployed, asked of the service", DIM)]))
    steps.append(("type", "curl -s $API/model-info"))
    steps.append(("out", [
        (f'  "model_name": "{info["model_name"]}",', STRING),
        (f'  "bundle_version": "{info["bundle_version"]}",'
         f'  "feature_version": {info["feature_version"]},', STRING),
        (f'  "decision_threshold": {info["decision_threshold"]:.3e},', STRING),
        (f'  "expected_precision": {info["expected_precision"]:.4f},', STRING),
        (f'  "precision_ceiling":  {info["precision_ceiling"]:.4f},', STRING),
        (f'  "dest_state_rows": {info["dest_state_rows"]:,},'
         f'  "snapshot_step": {info["dest_state_snapshot_step"]}', STRING),
    ]))
    steps.append(("hold", HOLD_SHORT))

    steps.append(("out", [("", FG), ("# a real transaction: origin drained to zero", DIM)]))
    steps.append(("type", "curl -s -X POST $API/score -d @fraud.json"))
    # .8f, not .6f: the true probability is 0.99999998..., and six places round
    # it to a flat 1.000000 -- overstating the model in its own demo.
    steps.append(("out", [
        (f'  "decision": "{fraud["decision"]}",  "flagged": {str(fraud["flagged"]).lower()},',
         ALERT if fraud["flagged"] else OK),
        (f'  "probability": {fraud["probability"]:.8f},', STRING),
        (f'  "decision_threshold": {fraud["decision_threshold"]:.3e},', STRING),
        (f'  "state_hit": {str(fraud["state_hit"]).lower()},'
         f'  "latency_ms": {fraud["latency_ms"]:.2f},', STRING),
        ('  "reasons": [', STRING),
    ] + [
        (f'     {r["feature"]:<28} {r["contribution"]:+.2f}', ACCENT)
        for r in fraud["reasons"][:3]
    ] + [
        ("  ]", STRING),
    ]))
    steps.append(("hold", HOLD_LONG))

    steps.append(("out", [("", FG), ("# and an ordinary payment, for contrast", DIM)]))
    steps.append(("type", "curl -s -X POST $API/score -d @payment.json"))
    steps.append(("out", [
        (f'  "decision": "{legit["decision"]}",  "flagged": {str(legit["flagged"]).lower()},',
         ALERT if legit["flagged"] else OK),
        (f'  "probability": {legit["probability"]:.3e},', STRING),
        (f'  "state_hit": {str(legit["state_hit"]).lower()}', STRING),
        ("       ^ no snapshot history, so cold-start defaults -- and it says so", DIM),
    ]))
    steps.append(("hold", HOLD_SHORT))

    steps.append(("out", [
        ("", FG),
        (f'# {cap["t_fraud"] * 1000:.0f} ms round trip, warm. Every response carries', DIM),
        ("# the threshold that produced it, so an alert stays auditable.", DIM),
    ]))
    steps.append(("hold", HOLD_LONG))
    return steps


def render(steps: list[tuple], output: Path) -> None:
    font = _load_font()
    char_w = font.getlength("M")
    line_h = FONT_SIZE + LINE_GAP
    width = int(PAD * 2 + char_w * COLS)

    # Two passes: measure the tallest the session ever gets, then render every
    # frame at that size. A GIF whose canvas changes mid-animation is invalid.
    total_lines = sum(1 if kind == "type" else len(payload) if kind == "out" else 0
                      for kind, payload in steps)
    height = int(PAD * 2 + line_h * (total_lines + 1))

    frames: list[Image.Image] = []
    lines: list[tuple[str, tuple]] = []

    def paint(partial: str | None = None) -> Image.Image:
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        y = PAD
        for text, colour in lines:
            d.text((PAD, y), text, font=font, fill=colour)
            y += line_h
        if partial is not None:
            d.text((PAD, y), "$ ", font=font, fill=PROMPT)
            d.text((PAD + char_w * 2, y), partial, font=font, fill=FG)
            d.text((PAD + char_w * (2 + len(partial)), y), "\u2588", font=font, fill=PROMPT)
        return img

    for kind, payload in steps:
        if kind == "type":
            for i in range(0, len(payload) + 1, TYPE_CHARS_PER_FRAME):
                frames.append(paint(payload[:i]))
            frames.append(paint(payload))
            lines.append(("$ " + payload, FG))
        elif kind == "out":
            for text, colour in payload:
                lines.append((text, colour))
                frames.append(paint())
        elif kind == "hold":
            frames.extend([frames[-1] if frames else paint()] * payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=FRAME_MS,
        loop=0, optimize=True, disposal=1,
    )
    size_kb = output.stat().st_size / 1024
    print(f"wrote {output} -- {len(frames)} frames, {width}x{height}, {size_kb:.0f} KB")
    if size_kb > SIZE_WARN_KB:
        print(f"  WARNING: over {SIZE_WARN_KB} KB for a README image. "
              "Reduce COLS, FONT_SIZE or the hold lengths.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-base", default=DEFAULT_API)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    render(build_script(capture(args.api_base)), args.output)


if __name__ == "__main__":
    main()
