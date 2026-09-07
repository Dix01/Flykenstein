"""Build the watchable page from an arcade run: telemetry + meta + video."""
import argparse, base64, json, sys
from pathlib import Path

import pandas as pd

KEEP = ["t_s","x","y","energy","hunger","arousal","spikes","active","kc","kc_active",
        "mbon","cx","depressed","da_max","dn_left","dn_right","dn_stop","mn9",
        "drive_l","drive_r","modulation","bump_strength","on_food","on_hazard"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=str)
    ap.add_argument("--out", type=str, default="/tmp/flykenstein_arcade.html")
    ap.add_argument("--template", type=str, default="scripts/arcade_template.html")
    a = ap.parse_args()

    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text())
    df = pd.read_csv(run / "telemetry.csv")
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols].copy()
    for c in ("on_food", "on_hazard"):
        if c in df:
            df[c] = df[c].astype(int)
    df = df.round(4)

    # Thin to about 12 samples per simulated second; the trace does not need more.
    step = max(1, int(len(df) / max(meta["seconds"] * 12, 1)))
    thin = df.iloc[::step].reset_index(drop=True)

    payload = {"meta": meta,
               "cols": list(thin.columns),
               "rows": thin.values.tolist(),
               "full_n": int(len(df))}

    video = run / "release.mp4"
    if video.exists():
        payload["video"] = ("data:video/mp4;base64,"
                            + base64.b64encode(video.read_bytes()).decode())

    html = Path(a.template).read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(payload))
    Path(a.out).write_text(html)
    mb = Path(a.out).stat().st_size / 1e6
    print(f"wrote {a.out}  ({mb:.1f} MB, {len(thin)} trace samples, "
          f"video {'embedded' if 'video' in payload else 'none'})")


if __name__ == "__main__":
    main()
