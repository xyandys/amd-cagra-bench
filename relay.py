#!/usr/bin/env python3
"""relay.py -- remote exec relay over the public bench repo.

Polls relay_cmd.txt every 5 s; when it changes, executes it with sh -c and
writes the combined output back to relay_out.txt.  Only repo writers (the
GHTOKEN holder) can change relay_cmd.txt, so the control channel is private.

Run:  GHTOKEN=ghp_xxx python3 relay.py
"""
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

REPO = "xyandys/amd-cagra-bench"
BRANCH = "main"
CMD_FILE = "relay_cmd.txt"
OUT_FILE = "relay_out.txt"
API = "https://api.github.com/repos/%s/contents" % REPO
TOKEN = os.environ.get("GHTOKEN", "")
HDRS = {
    "Authorization": "token " + TOKEN,
    "Accept": "application/vnd.github+json",
    "User-Agent": "amd-cagra-relay",
}


def gh_json(url, data=None, method=None):
    if data is not None:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body, method=method or "PUT",
            headers=dict(HDRS, **{"Content-Type": "application/json"}))
    else:
        req = urllib.request.Request(url, headers=HDRS, method=method)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode() or "{}")


def get_file(path):
    try:
        j = gh_json("%s/%s?ref=%s&t=%d" % (API, path, BRANCH, time.time() * 1000))
        return base64.b64decode(j["content"]).decode("utf-8", "replace"), j.get("sha")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, None
        raise


def put_file(path, text, sha, msg):
    data = {
        "message": msg,
        "branch": BRANCH,
        "content": base64.b64encode(text.encode()).decode(),
    }
    if sha:
        data["sha"] = sha
    gh_json("%s/%s" % (API, path), data, method="PUT")


def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
    out = "$ %s\n[rc=%d]\n--- stdout ---\n%s\n--- stderr ---\n%s\n" % (
        cmd, p.returncode, p.stdout, p.stderr)
    return out, p.returncode


def main():
    print("relay: cwd=%s pid=%d" % (os.getcwd(), os.getpid()), flush=True)
    print("relay: polling %s every 5 s ..." % CMD_FILE, flush=True)
    last = None
    first = True
    while True:
        try:
            cmd, _ = get_file(CMD_FILE)
            if cmd is not None:
                if first:
                    last = cmd  # adopt stale content, do not execute
                elif cmd != last:
                    last = cmd
                    print("relay: exec >>> %r" % cmd, flush=True)
                    try:
                        out, rc = run_cmd(cmd)
                    except subprocess.TimeoutExpired as e:
                        out = "$ %s\n[TIMEOUT after 3600s]\n%s\n" % (
                            cmd, (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
                        rc = -1
                    out = out[-200000:]
                    _, sha = get_file(OUT_FILE)
                    put_file(OUT_FILE, out, sha,
                             "relay out %s" % time.strftime("%H:%M:%S"))
                    print("relay: rc=%d, %d bytes -> %s" % (rc, len(out), OUT_FILE), flush=True)
            first = False
        except Exception as e:
            print("relay: error: %r" % (e,), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
