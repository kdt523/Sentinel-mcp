"""Mock website with benign and poisoned pages (indirect-injection test bed).

Pages are listed in ``manifest.yaml``. The manifest's labels are ground truth for tests and
dataset building and are deliberately NOT exposed over HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from tools_servers.content_gen import ContentIdError, generate_page

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.yaml"
PAGES_DIR = HERE / "pages"


@dataclass(frozen=True)
class Page:
    slug: str
    file: str
    label: str
    technique: str | None = None
    target: str | None = None
    hard_negative: bool = False

    @property
    def html(self) -> str:
        return (PAGES_DIR / self.file).read_text(encoding="utf-8")


def load_pages(manifest: Path = MANIFEST) -> dict[str, Page]:
    data: dict[str, Any] = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    pages = {p["slug"]: Page(**p) for p in data["pages"]}
    for page in pages.values():
        if page.label not in {"benign", "poisoned"}:
            raise ValueError(f"bad label for {page.slug}: {page.label}")
        if not (PAGES_DIR / page.file).is_file():
            raise FileNotFoundError(page.file)
    return pages


def create_app() -> FastAPI:
    pages = load_pages()
    app = FastAPI(title="mockweb", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return {"service": "acme-notes", "status": "operational", "incidents": []}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        links = "\n".join(f'<li><a href="/{slug}">{slug}</a></li>' for slug in sorted(pages))
        return f"<!doctype html><html><body><h1>Site index</h1><ul>\n{links}\n</ul></body></html>"

    @app.get("/g/{item_id}", response_class=HTMLResponse)
    async def generated(item_id: str) -> str:
        """Generated poisoned / decoy pages (tools_servers/content_gen.py)."""
        try:
            return generate_page(item_id).text
        except ContentIdError as exc:
            raise HTTPException(status_code=404, detail="page not found") from exc

    @app.get("/{slug:path}", response_class=HTMLResponse)
    async def page(slug: str) -> str:
        found = pages.get(slug.strip("/"))
        if found is None:
            raise HTTPException(status_code=404, detail="page not found")
        return found.html

    return app


app = create_app()
