from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import omni.client

ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

def list_dir(url):
    print(f"\n=== {url} ===")
    result, entries = omni.client.list(url)
    print("result:", result)
    if entries:
        for e in entries:
            print(e.relative_path)

for path in [
    ROOT,
    ROOT + "/Isaac",
    ROOT + "/Isaac/Robots",
    ROOT + "/Isaac/People",
    ROOT + "/Isaac/Environments",
    ROOT + "/Isaac/Props",
    ROOT + "/NVIDIA",
]:
    try:
        list_dir(path)
    except Exception as e:
        print(path, "ERROR:", repr(e))

simulation_app.close()
