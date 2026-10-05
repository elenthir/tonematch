"""TONE3000 client: OAuth 2.0 + PKCE login (loopback redirect), tone search, capture / IR download.

Every TONE3000 API call needs a user login. Once, on tone3000.com → Settings → API Keys, create a key
(the publishable key ``t3k_pub_…`` is the OAuth client id) and register the redirect URI
``http://localhost:3927/callback``. Then ``tonematch tone3000 login --client-id t3k_pub_…`` opens the
browser; tokens are kept in ``~/.tonematch/tone3000.json`` and refreshed automatically.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .catalog import IR_DIR, NAM_DIR, TONEMATCH_HOME

BASE = "https://www.tone3000.com/api/v1"
AUTHORIZE_URL = "https://www.tone3000.com/api/v1/oauth/authorize"
TOKEN_URL = BASE + "/oauth/token"
REDIRECT_PORT = 3927
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"
TOKEN_FILE = TONEMATCH_HOME / "tone3000.json"
GEARS = ("amp", "amp-cab", "full-rig", "pedal", "outboard", "cab", "ir")
SIZES = ("standard", "lite", "feather", "nano", "custom")
SORTS = ("best-match", "newest", "oldest", "trending", "downloads-all-time")
USER_AGENT = "tonematch/0.1 (+https://github.com/elenthir/tonematch)"


class Tone3000Error(SystemExit):
    pass


@dataclass
class Tokens:
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: float = 0.0
    client_id: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:60] or "tone"


# --------------------------------------------------------------------------- transport
Transport = Callable[[str, str, Optional[bytes], Dict[str, str]], "Response"]


@dataclass
class Response:
    status: int
    body: bytes
    headers: Dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8") or "null")


def urllib_transport(method: str, url: str, data: Optional[bytes], headers: Dict[str, str]) -> Response:
    req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return Response(r.status, r.read(), {k.lower(): v for k, v in r.headers.items()})
    except urllib.error.HTTPError as e:
        return Response(e.code, e.read() or b"", {k.lower(): v for k, v in e.headers.items()})


# --------------------------------------------------------------------------- client
class Tone3000:
    def __init__(self, tokens: Optional[Tokens] = None, transport: Transport = urllib_transport,
                 token_file: Path = TOKEN_FILE):
        self.tokens = tokens
        self.transport = transport
        self.token_file = token_file
        if self.tokens is None and token_file.exists():
            try:
                self.tokens = Tokens(**json.loads(token_file.read_text()))
            except Exception:
                self.tokens = None

    # ----------------------------------------------------------------- auth
    @staticmethod
    def pkce() -> tuple:
        verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=").decode()
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        return verifier, challenge

    @staticmethod
    def authorize_url(client_id: str, challenge: str, state: str, redirect_uri: str = REDIRECT_URI) -> str:
        q = {"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
             "code_challenge": challenge, "code_challenge_method": "S256", "state": state}
        return AUTHORIZE_URL + "?" + urllib.parse.urlencode(q)

    def _token_request(self, form: Dict[str, str], client_id: str) -> Tokens:
        body = urllib.parse.urlencode(form).encode()
        r = self.transport("POST", TOKEN_URL, body, {"Content-Type": "application/x-www-form-urlencoded",
                                                      "Accept": "application/json"})
        if r.status >= 400:
            raise Tone3000Error(f"TONE3000 token request failed ({r.status}): {r.body[:300].decode(errors='replace')}")
        d = r.json()
        t = Tokens(access_token=d["access_token"], refresh_token=d.get("refresh_token"),
                   expires_at=time.time() + float(d.get("expires_in") or 3600), client_id=client_id)
        self.tokens = t
        self._save()
        return t

    def _save(self) -> None:
        if self.tokens is None:
            return
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(json.dumps(self.tokens.to_dict()))
        try:
            os.chmod(self.token_file, 0o600)
        except OSError:
            pass

    def exchange_code(self, code: str, verifier: str, client_id: str, redirect_uri: str = REDIRECT_URI) -> Tokens:
        return self._token_request({"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
                                    "client_id": client_id, "code_verifier": verifier}, client_id)

    def refresh(self) -> Tokens:
        if not self.tokens or not self.tokens.refresh_token:
            raise Tone3000Error("not logged in to TONE3000 — run: tonematch tone3000 login --client-id t3k_pub_…")
        return self._token_request({"grant_type": "refresh_token", "refresh_token": self.tokens.refresh_token,
                                    "client_id": self.tokens.client_id}, self.tokens.client_id)

    def login(self, client_id: str, open_browser: bool = True, timeout: float = 300.0,
              log: Callable[[str], None] = print) -> Tokens:
        """Loopback OAuth: open the browser, catch the redirect on localhost, exchange the code."""
        verifier, challenge = self.pkce()
        state = secrets.token_urlsafe(16)
        result: Dict[str, str] = {}
        done = threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
                result.update(q)
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h2>tonematch: you can close this tab.</h2>")
                done.set()

            def log_message(self, *a):  # silence
                pass

        srv = http.server.HTTPServer(("127.0.0.1", REDIRECT_PORT), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = self.authorize_url(client_id, challenge, state)
        log(f"open this URL in your browser if it did not open by itself:\n  {url}")
        if open_browser:
            webbrowser.open(url)
        try:
            if not done.wait(timeout):
                raise Tone3000Error("login timed out")
        finally:
            srv.shutdown()
        if result.get("state") != state:
            raise Tone3000Error("OAuth state mismatch — try again")
        if result.get("error") or result.get("canceled"):
            raise Tone3000Error(f"login refused: {result.get('error') or 'canceled'}")
        return self.exchange_code(result["code"], verifier, client_id)

    # ----------------------------------------------------------------- requests
    def _auth_get(self, url: str, accept: str = "application/json") -> Response:
        if not self.tokens:
            raise Tone3000Error("not logged in to TONE3000 — run: tonematch tone3000 login --client-id t3k_pub_…")
        if self.tokens.expires_at and time.time() > self.tokens.expires_at - 60 and self.tokens.refresh_token:
            self.refresh()
        for attempt in (0, 1):
            r = self.transport("GET", url, None, {"Authorization": f"Bearer {self.tokens.access_token}", "Accept": accept})
            if r.status == 401 and attempt == 0 and self.tokens.refresh_token:
                self.refresh()
                continue
            if r.status == 429:
                wait = float(r.headers.get("retry-after", "5") or 5)
                time.sleep(min(wait, 30))
                continue
            break
        if r.status >= 400:
            raise Tone3000Error(f"TONE3000 request failed ({r.status}) {url}: {r.body[:300].decode(errors='replace')}")
        return r

    def get(self, path: str, **params) -> Any:
        q = {k: v for k, v in params.items() if v not in (None, "", [], ())}
        url = BASE + path + ("?" + urllib.parse.urlencode(q) if q else "")
        return self._auth_get(url).json()

    def search(self, query: str = "", gears: Optional[List[str]] = None, sizes: Optional[List[str]] = None,
               fmt: Optional[str] = "nam", sort: str = "best-match", page: int = 1, page_size: int = 20,
               architecture: str = "1", tags: Optional[List[str]] = None, makes: Optional[List[str]] = None) -> dict:
        return self.get("/tones/search", query=query or None, gears="_".join(gears or []) or None,
                        sizes="_".join(sizes or []) or None, format=fmt, sort=sort, page=page, page_size=page_size,
                        architecture=architecture, tags="_".join(tags or []) or None, makes="_".join(makes or []) or None)

    def tone(self, tone_id: int) -> dict:
        return self.get(f"/tones/{tone_id}")

    def models(self, tone_id: int, architecture: str = "1", page_size: int = 50) -> List[dict]:
        d = self.get("/models", tone_id=tone_id, architecture=architecture, page_size=page_size)
        return d.get("data", d) if isinstance(d, dict) else d

    def download(self, model_url: str) -> bytes:
        return self._auth_get(model_url, accept="*/*").body

    # ----------------------------------------------------------------- high level
    def fetch(self, query: str, limit: int = 10, gears: Optional[List[str]] = None,
              sizes: Optional[List[str]] = None, sort: str = "downloads-all-time", nam_dir: Path = NAM_DIR,
              ir_dir: Path = IR_DIR, prefer_sizes=("standard", "lite", "feather", "nano"),
              with_irs: bool = True, log: Callable[[str], None] = print) -> List[Path]:
        """Search, pick one model per tone (best size available), download captures (+ IRs) into
        the library the catalog reads. Returns the written files."""
        written: List[Path] = []
        fetched_tones = 0
        page = 1
        while fetched_tones < limit:
            res = self.search(query, gears=gears or ["amp", "amp-cab", "full-rig"], sizes=sizes, fmt="nam",
                              sort=sort, page=page, page_size=min(50, limit))
            tones = res.get("data", [])
            if not tones:
                break
            for t in tones:
                if fetched_tones >= limit:
                    break
                try:
                    f = self._fetch_tone(t, nam_dir, prefer_sizes, log)
                except Tone3000Error as e:
                    log(f"  skip {t.get('title')}: {e}")
                    continue
                if f:
                    written.append(f)
                    fetched_tones += 1
            if page >= int(res.get("total_pages") or 1):
                break
            page += 1
        if with_irs and not any(ir_dir.rglob("*.wav")):
            try:
                res = self.search(query, gears=["cab", "ir"], fmt="ir", sort=sort, page_size=10)
                for t in res.get("data", [])[:5]:
                    f = self._fetch_tone(t, ir_dir, prefer_sizes, log, is_ir=True)
                    if f:
                        written.append(f)
            except Tone3000Error as e:
                log(f"  IR search failed: {e}")
        return written

    def _fetch_tone(self, tone: dict, root: Path, prefer_sizes, log, is_ir: bool = False) -> Optional[Path]:
        tid = tone.get("id")
        title = tone.get("title") or f"tone{tid}"
        folder = root / f"{tid}-{_slug(title)}"
        existing = list(folder.glob("*.wav" if is_ir else "*.nam")) if folder.exists() else []
        if existing:
            log(f"  have   {title}")
            return existing[0]
        models = self.models(tid, architecture="1")
        if not models:
            log(f"  skip   {title}: no downloadable model")
            return None
        rank = {s: i for i, s in enumerate(prefer_sizes)}
        models.sort(key=lambda m: rank.get(str(m.get("size") or ""), 99))
        m = models[0]
        folder.mkdir(parents=True, exist_ok=True)
        name = _slug(m.get("name") or title)
        if is_ir:
            if not name.lower().endswith((".wav", ".aif", ".aiff")):
                name += ".wav"
        elif not name.lower().endswith(".nam"):
            name += ".nam"
        data = self.download(m["model_url"])
        dst = folder / name
        dst.write_bytes(data)
        side = {"tone_id": tid, "title": title, "url": tone.get("url"), "gear": tone.get("gear"),
                "make": ", ".join(x.get("name", "") for x in tone.get("makes") or []),
                "tags": [x.get("name", "") for x in tone.get("tags") or []],
                "username": (tone.get("user") or {}).get("username"), "license": tone.get("license"),
                "downloads_count": tone.get("downloads_count"), "favorites_count": tone.get("favorites_count"),
                "size": m.get("size"), "model_id": m.get("id"), "model_name": m.get("name")}
        (folder / (dst.name + ".json")).write_text(json.dumps(side, indent=1))
        log(f"  got    {title}  [{m.get('size')}] → {dst}")
        return dst
