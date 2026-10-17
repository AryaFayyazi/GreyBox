"""Hash-lock a preregistration: python scripts/lock.py <name> <file> [<file> ...] -> results/prereg_<name>_lock.json"""
import datetime, hashlib, json, sys
from pathlib import Path
name, files = sys.argv[1], sys.argv[2:]
out = {"locked_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
       "sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
p = Path("/home/arya/graybox/results") / f"prereg_{name}_lock.json"
assert not p.exists(), f"{p} exists: a lock is never overwritten"
p.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
