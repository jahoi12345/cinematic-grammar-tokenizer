"""HTTP verification using a scratch root, synthetic media and scratch labels only."""

import http.client
import http.server
import importlib.util
import json
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import cv2
import numpy as np
import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def label_module(tmp_path):
    spec = importlib.util.spec_from_file_location("scratch_label_server", PROJECT / "app/label/server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "films.yaml").write_text(yaml.safe_dump({"films": [{
        "id": "night_of_the_living_dead", "path": "synthetic.mp4"
    }]}))
    # The range handler only transports bytes; no real film is needed.
    (tmp_path / "synthetic.mp4").write_bytes(bytes(range(256)) * 32)
    interim = tmp_path / "data/interim/night_of_the_living_dead"
    interim.mkdir(parents=True)
    (interim / "cuts_t12.json").write_text(json.dumps([[0, 2], [2, 5], [5, 6]]))
    assert cv2.imwrite(str(interim / "synthetic.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
    (tmp_path / "pyproject.toml").write_text("DO NOT SERVE THIS FILE")
    return module


@contextmanager
def running_server(module):
    """Match CLI cuts-mode initialization, with a new server instance on each entry."""
    items, path = module.load_items("cuts", SimpleNamespace(
        film="night_of_the_living_dead", threshold=12, n=2
    ))
    module.S.clear()
    module.S.update(mode="cuts", items=items, path=path,
                    labels=json.loads(path.read_text()) if path.exists() else {})
    errors = []

    class Server(http.server.ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            errors.append(traceback.format_exc())

    server = Server(("127.0.0.1", 0), module.H)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], errors
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(port, path, method="GET", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_image_traversal_denied_and_valid_image_served(label_module):
    with running_server(label_module) as (port, errors):
        bad = "/img?" + urlencode({"p": "data/interim/../../pyproject.toml"})
        assert request(port, bad)[0] == 403
        valid = "/img?" + urlencode({"p": "data/interim/night_of_the_living_dead/synthetic.jpg"})
        status, headers, body = request(port, valid)
        assert status == 200
        assert headers["Content-Type"] == "image/jpeg"
        assert body.startswith(b"\xff\xd8")
        assert not errors


def test_range_returns_206_and_exact_1001_bytes(label_module):
    with running_server(label_module) as (port, errors):
        status, headers, body = request(port, "/video?film=night_of_the_living_dead",
                                        headers={"Range": "bytes=1000-2000"})
        assert status == 206
        assert headers["Content-Length"] == "1001"
        assert headers["Content-Range"] == "bytes 1000-2000/8192"
        assert body == (label_module.ROOT / "synthetic.mp4").read_bytes()[1000:2001]
        assert len(body) == 1001
        assert not errors


@pytest.mark.parametrize("film", ["unknown_film", "../../pyproject.toml"])
def test_bad_film_returns_client_error_and_server_survives(label_module, film):
    with running_server(label_module) as (port, errors):
        try:
            status, _, body = request(port, "/video?" + urlencode({"film": film}))
        except http.client.RemoteDisconnected:
            status, body = None, b""
        assert request(port, "/api/state")[0] == 200, "Server must remain responsive"
        assert b"DO NOT SERVE THIS FILE" not in body
        assert status in (400, 403, 404), f"Expected controlled client error, got {status}; errors={errors}"
        assert not errors, "Bad IDs must not raise uncaught request-handler exceptions"


def test_label_survives_server_restart_with_atomic_replace(label_module, monkeypatch):
    replacements = []
    original_replace = Path.replace

    def tracked_replace(source, target):
        replacements.append((source, target, source.read_text()))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", tracked_replace)
    with running_server(label_module) as (port, errors):
        state = json.loads(request(port, "/api/state")[2])
        item_id = state["items"][0]["id"]
        body = json.dumps({"id": item_id, "label": "real"})
        assert request(port, "/api/label", "POST", body)[0] == 200
        label_path = label_module.S["path"]
        assert json.loads(label_path.read_text()) == {item_id: "real"}
        assert replacements == [(label_path.with_suffix(".tmp"), label_path,
                                  label_path.read_text())]
        assert not label_path.with_suffix(".tmp").exists()
        assert not errors
    with running_server(label_module) as (port, errors):
        assert json.loads(request(port, "/api/state")[2])["labels"] == {item_id: "real"}
        assert not errors


def test_fields_follow_schema_and_documented_extras(label_module):
    schema = yaml.safe_load((PROJECT / "configs/schema.yaml").read_text())["attributes"]
    fields = {key: values for key, _, values in label_module.FIELDS}
    for key in ("size", "dof", "key"):
        assert fields[key] == schema[key]["values"]
    assert fields["angle"] == ["level", *schema["angle"]["values"]]
    assert fields["people"] == [str(n) for n in range(schema["people"]["max"] + 1)]
    assert fields["facing"] == ["away", "camera", "profile"]
    assert fields["light_dir"] == ["front", "side", "back"]
