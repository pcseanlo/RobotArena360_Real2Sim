"""Optional tool: request / download a World Labs Marble world for a scene.

One splat provider among possible others (``r2s/world/providers/``); not a
pipeline step. The splats in use were downloaded from the Marble web UI as PLY
(2.2M gaussians, SH degree 1) and are imported by ``world.import``. This client
is the cleaned-up version of the old repo's Marble script, for doing the same
through the API:

    export WLT_API_KEY=...                       # never stored in a file
    python -m r2s.world.providers.marble generate --scene droid2   # upload background.png, start generation
    python -m r2s.world.providers.marble status   --scene droid2   # poll; records world_id when done
    python -m r2s.world.providers.marble download --scene droid2   # assets -> <run>/scenes/<s>/world/marble/

Fixes over the old script: the key only comes from ``WLT_API_KEY``; the record is
one JSON per scene (``world/marble.json``) instead of global mapping files that each
run truncated (``w+``); ``world_id`` is read from the finished operation instead of
being saved as null; downloads use ``requests`` instead of ``os.system(curl)``.

The API delivers ``.spz`` splats; ``world.import`` needs a 3DGS ``.ply``. Convert
with a SPZ tool (e.g. nianticlabs/spz) or download the PLY from the web UI.
Untested against the live API in this repo (no key was used while porting).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from r2s.common.io import prepare_output, write_json
from r2s.layout import SceneLayout

API = "https://api.worldlabs.ai/marble/v1"
MODEL = "marble-1.1"


class Marble:
    def __init__(self, api_key: str):
        import requests

        self.requests = requests
        self.headers = {"WLT-Api-Key": api_key}

    def _post(self, path: str, payload: dict) -> dict:
        r = self.requests.post(f"{API}/{path}", headers={**self.headers, "Content-Type": "application/json"}, json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"POST {path} failed ({r.status_code}): {r.text}")
        return r.json()

    def _get(self, path: str) -> dict:
        r = self.requests.get(f"{API}/{path}", headers=self.headers, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"GET {path} failed ({r.status_code}): {r.text}")
        return r.json()

    def upload_image(self, image: Path, name: str) -> str:
        info = self._post("media-assets:prepare_upload", {"file_name": name, "kind": "image", "extension": image.suffix.lstrip(".")})
        up = info["upload_info"]
        r = self.requests.put(up["upload_url"], data=image.read_bytes(), headers=up["required_headers"], timeout=300)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"image upload failed ({r.status_code}): {r.text}")
        return info["media_asset"]["media_asset_id"]

    def generate(self, media_asset_id: str, display_name: str, text_prompt: str | None = None) -> dict:
        payload = {
            "display_name": display_name,
            "world_prompt": {"type": "image", "image_prompt": {"source": "media_asset", "media_asset_id": media_asset_id}},
            "model": MODEL,
        }
        if text_prompt:
            payload["world_prompt"]["text_prompt"] = text_prompt
        return self._post("worlds:generate", payload)

    def operation(self, operation_id: str) -> dict:
        return self._get(f"operations/{operation_id}")

    def world(self, world_id: str) -> dict:
        return self._get(f"worlds/{world_id}")

    def download(self, url: str, dst: Path) -> None:
        with self.requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with prepare_output(dst).open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)


def _record_path(layout: SceneLayout) -> Path:
    return layout.world_dir / "marble.json"


def _load_record(layout: SceneLayout) -> dict:
    path = _record_path(layout)
    return json.loads(path.read_text()) if path.exists() else {"scene": layout.scene}


def _world_id_from(op: dict) -> str | None:
    for container in (op.get("response") or {}, op.get("metadata") or {}):
        for key in ("world_id", "id"):
            if container.get(key):
                return str(container[key])
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "status", "download"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prompt", default=None, help="optional text prompt for generate")
    args = parser.parse_args(argv)

    key = os.environ.get("WLT_API_KEY")
    if not key:
        print("set WLT_API_KEY in the environment", file=sys.stderr)
        return 2
    layout = SceneLayout.create(args.scene)
    marble = Marble(key)
    record = _load_record(layout)

    if args.command == "generate":
        image = layout.inputs.background_png
        if "media_asset_id" not in record:
            record["media_asset_id"] = marble.upload_image(image, f"{layout.scene}.png")
            record["image"] = str(image)
        resp = marble.generate(record["media_asset_id"], layout.scene, args.prompt)
        record["operation_id"] = resp["operation_id"]
        record["world_id"] = _world_id_from(resp)
        record["model"] = MODEL
    elif args.command == "status":
        op = marble.operation(record["operation_id"])
        record["done"] = bool(op.get("done"))
        record["world_id"] = _world_id_from(op) or record.get("world_id")
        print(json.dumps({k: record.get(k) for k in ("done", "world_id")}))
    else:
        if not record.get("world_id"):
            print("no world_id yet; run `status` after generation finishes", file=sys.stderr)
            return 1
        assets = marble.world(record["world_id"])["assets"]
        out = layout.world_dir / "marble"
        for name, url in assets["splats"]["splat_urls"].items():
            marble.download(url, out / f"splat_{name}.spz")
        marble.download(assets["mesh"]["collider_mesh_url"], out / "collider_mesh.glb")
        record["downloaded_to"] = str(out)
    write_json(_record_path(layout), record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
