"""Single-command launch serves the scientific API and compiled interface."""
from fastapi.testclient import TestClient
import run


def test_launcher_serves_ui_assets_and_scientific_api():
    run.prepare_frontend()
    with TestClient(run.create_app()) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "OncoEvo Research Studio" in response.text
        import re
        script = re.search(r'src="([^"]+\.js)"', response.text).group(1)
        assert client.get(script).status_code == 200
        status = client.get("/api/status").json()
        assert status["cohort_size"] == 71 and status["model"]["available"]


def test_launcher_rebuilds_when_source_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "FRONTEND", tmp_path)
    (tmp_path / "src").mkdir()
    assert run.frontend_needs_build()
    for name in ("package.json", "package-lock.json", "index.html", "vite.config.ts", "tsconfig.json", "tailwind.config.js", "postcss.config.js"):
        (tmp_path / name).write_text("")
    (tmp_path / "dist").mkdir()
    built = tmp_path / "dist/index.html"
    built.write_text("built")
    import os
    os.utime(built, (2000000000, 2000000000))
    assert not run.frontend_needs_build()
    source = tmp_path / "src/App.tsx"
    source.write_text("changed")
    os.utime(source, (2000000001, 2000000001))
    assert run.frontend_needs_build()
