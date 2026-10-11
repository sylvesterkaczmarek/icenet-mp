"""Download public OSI SAF daily SIC files for time-separated x0-scaling evaluation.

No credentials are read or sent. Files and manifest remain outside the repository.
"""
import concurrent.futures
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

DEST = Path("/Volumes/T7/Research-Data/alan-turing-institute/cryocast/x0scale-478-expanded-20261011")
BASE = "ftp://osisaf.met.no/reprocessed/ice/conc-cont-reproc-amsr/v3p0"
SPANS = {
    "training": (dt.date(2023, 1, 1), dt.date(2023, 4, 30)),
    "validation": (dt.date(2024, 1, 1), dt.date(2024, 1, 31)),
    "testing": (dt.date(2024, 2, 1), dt.date(2024, 2, 29)),
}
def dates_inclusive(start, end):
    return [start + dt.timedelta(days=i) for i in range((end-start).days+1)]

def get_one(item):
    split, date = item
    file = f"ice_conc_nh_ease2-250_icdr-v3p0-amsr_{date:%Y%m%d}1200.nc"
    destination = DEST / "raw" / split / file
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{date:%Y/%m}/{file}"
    if not (destination.exists() and destination.stat().st_size > 50000):
        part = destination.with_suffix(".part")
        cmd = ["curl","-fL","--retry","4","--retry-delay","1","--connect-timeout","20","--max-time","100","--silent","--show-error","--output",str(part),url]
        subprocess.run(cmd,check=True)
        if part.stat().st_size < 50000:
            raise RuntimeError(f"Truncated file {url}")
        part.replace(destination)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"split":split,"date":date.isoformat(),"file":str(destination.relative_to(DEST)),"sha256":digest,"bytes":destination.stat().st_size,"source":url}

def main():
    entries=[(name,d) for name,(a,b) in SPANS.items() for d in dates_inclusive(a,b)]
    assert len(entries)==180
    DEST.mkdir(parents=True,exist_ok=True)
    print("START",len(entries),"public OSI SAF NetCDF files",flush=True)
    start=time.monotonic()
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(get_one,item) for item in entries]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            results.append(f.result())
            if i%15==0 or i==len(entries):
                print("DOWNLOADED",i,len(entries),"elapsed_s",round(time.monotonic()-start,1),flush=True)
    results.sort(key=lambda r:(r["date"],r["split"]))
    assert len({r["date"] for r in results})==len(results)
    (DEST/"download_manifest.json").write_text(json.dumps({"source":BASE,"spans":{k:[str(a),str(b)] for k,(a,b) in SPANS.items()},"files":results},indent=2))
    print("COMPLETE",len(results),"files",sum(r["bytes"] for r in results),"bytes, seconds",round(time.monotonic()-start,1),flush=True)
if __name__=="__main__":
    main()
