"""Stage 2.5a — Generate benign REST API traffic programmatically."""
from __future__ import annotations
import argparse, json, random, uuid
from pathlib import Path
import pyarrow.parquet as pq
from ai_waf_v2.data.schema import HttpRecord, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)

ENDPOINTS = [
    ("GET",  "/api/v1/users",            "page={p}&limit={l}"),
    ("GET",  "/api/v1/products",         "category={cat}&sort=price&order=asc"),
    ("POST", "/api/v1/auth/login",       ""),
    ("POST", "/api/v1/orders",           ""),
    ("GET",  "/api/v1/search",           "q={q}&page=1"),
    ("PUT",  "/api/v1/users/{id}",       ""),
    ("DELETE","/api/v1/cart/{id}",       ""),
    ("GET",  "/api/v1/categories",       ""),
    ("POST", "/api/v1/auth/refresh",     ""),
    ("GET",  "/api/v1/profile",          ""),
]
BENIGN_QUERIES = [
    "blue+shirt", "summer+sale", "size+medium",
    "best+sellers", "new+arrivals", "red+shoes",
    "organic+food", "laptop+stand", "office+chair",
]
CATEGORIES = ["electronics","clothing","furniture","books","sports","toys"]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "PostmanRuntime/7.36.0",
    "python-requests/2.31.0",
]

def make_record(rng: random.Random) -> HttpRecord:
    method, path, qs_tmpl = rng.choice(ENDPOINTS)
    path = path.replace("{id}", str(rng.randint(1, 9999)))
    qs = qs_tmpl.format(
        p=rng.randint(1,100), l=rng.choice([10,20,50]),
        cat=rng.choice(CATEGORIES), q=rng.choice(BENIGN_QUERIES),
    )
    body = ""
    headers = {
        "Host": "api.example.com",
        "User-Agent": rng.choice(USER_AGENTS),
        "Accept": "application/json",
        "Authorization": f"Bearer eyJ{uuid.uuid4().hex[:16]}",
    }
    if method in ("POST","PUT"):
        body = json.dumps({"key": str(uuid.uuid4())[:8], "value": rng.randint(1,100)})
        headers["Content-Type"] = "application/json"
    return HttpRecord(
        id=str(uuid.uuid4()), method=method, path=path,
        query_string=qs, headers=json.dumps(headers), body=body,
        label=0, attack_class="benign", source="aug_benign_rest",
    ).build_raw()

def run(args):
    configure_root(); cfg = load_config(args.config)
    rng = random.Random(cfg.project.seed)
    n = cfg.augmentation.benign.rest_samples
    records = [make_record(rng) for _ in range(n)]
    out = Path(cfg.paths.data_augmented)/"benign"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(records_to_table(records), out/"benign_rest.parquet", compression="snappy")
    log.info(f"Generated {len(records):,} benign REST samples")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
