#!/bin/sh
set -e
python - <<'PY'
import os
import sys
import time
import urllib.request

url = os.environ.get("QDRANT_URL", "http://qdrant:6333").rstrip("/") + "/collections"
deadline = time.time() + 90
last = None
while time.time() < deadline:
    try:
        urllib.request.urlopen(url, timeout=2)
        print("qdrant is reachable:", url, flush=True)
        sys.exit(0)
    except Exception as exc:
        last = exc
        time.sleep(1)
print("timeout waiting for qdrant:", url, last, file=sys.stderr)
sys.exit(1)
PY
exec "$@"
