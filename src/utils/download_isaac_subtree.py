from isaaclab.app import AppLauncher
import argparse
import shutil
import urllib.request
import urllib.parse
from pathlib import Path, PurePosixPath

parser = argparse.ArgumentParser()
parser.add_argument("subtree", help="Example: Isaac/Robots/Unitree")
parser.add_argument(
    "--local-root",
    default="~/isaacsim_assets_partial/Assets/Isaac/5.1",
    help="Local mirror root corresponding to Assets/Isaac/5.1",
)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import omni.client

ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
subtree = args.subtree.strip("/")
start_url = ROOT + "/" + subtree
local_root = Path(args.local_root).expanduser()

files = []

def is_dir(entry):
    try:
        return bool(entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)
    except Exception:
        return "." not in entry.relative_path

def walk(url):
    result, entries = omni.client.list(url)
    if result != omni.client.Result.OK or not entries:
        print("[WARN] cannot list:", url, "result:", result)
        return

    for e in entries:
        child_url = url.rstrip("/") + "/" + e.relative_path
        if is_dir(e):
            walk(child_url)
        else:
            files.append(child_url)

def download_file(url):
    rel = url.replace(ROOT + "/", "")
    out_path = local_root / Path(PurePosixPath(rel))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print("[SKIP]", rel)
        return

    print("[GET ]", rel)
    quoted_url = urllib.parse.quote(url, safe=":/?&=%")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with urllib.request.urlopen(quoted_url) as response, open(tmp_path, "wb") as f:
        shutil.copyfileobj(response, f)

    tmp_path.rename(out_path)

print("Remote subtree:", start_url)
print("Local root:", local_root)

walk(start_url)

print(f"Files found: {len(files)}")

for url in files:
    rel = url.replace(ROOT + "/", "")
    if args.dry_run:
        print("[DRY]", rel)
    else:
        download_file(url)

simulation_app.close()
