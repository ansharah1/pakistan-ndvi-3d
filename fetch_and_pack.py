"""
fetch_and_pack.py — display frames + elevation straight from Earth Engine -> pak_ndvi_frames.json
Then run build_map.py (statistics + final HTML).  Settings live in ndvi_common.py.
"""
import base64, json, math, os, time, warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import ee
import ndvi_common as C

warnings.filterwarnings("ignore", category=DeprecationWarning)
W = 700            # display grid width in cells (~2.9 km); display only, statistics use their own grid
NMAX = 0.8         # display colour/height top; values above shown as >= 0.8
WORKERS = 6
OUT = "pak_ndvi_frames.json"

C.init()
pak = C.boundary()
x0, y0, x1, y1 = C.bounds(pak)
res = (x1 - x0) / W
H = int(math.ceil((y1 - y0) / res))
y0 = y1 - H * res
GRID = {"dimensions": {"width": W, "height": H},
        "affineTransform": {"scaleX": res, "shearX": 0, "translateX": x0,
                            "shearY": 0, "scaleY": -res, "translateY": y1},
        "crsCode": "EPSG:4326"}
print(f"Display grid {W} x {H}, cell ≈ {res * 111:.1f} km")


def fetch(img, retries=4):
    img = ee.Image(img).rename("v").unmask(-9999, False).toFloat()
    for attempt in range(retries):
        try:
            arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": GRID})
            return np.asarray(arr["v"], dtype="float32")
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retrying after: {str(e)[:90]}")
            time.sleep(4 * (attempt + 1))


mask = fetch(ee.Image(1).clipToCollection(pak)) > 0.5
print(f"Boundary mask: {int(mask.sum())} display cells")

dem = fetch(ee.ImageCollection(C.DEM_ID).select("DEM").mosaic())       # pinned release, no silent fallback
dem_ok = mask & (dem > -1000)
dem = np.where(dem_ok, np.clip(dem, 0, 9000), 0)
dem_max = float(dem.max())
print(f"Elevation ({C.DEM_ID}): display-grid max {dem_max:.0f} m")

region = ee.Geometry.Rectangle([x0, y0, x1, y1], "EPSG:4326", False)
coll = C.ndvi_collection(region)
n = coll.size().getInfo()
if n == 0:
    raise SystemExit("No NDVI composites in the date range.")
lst = coll.toList(n)
meta = ee.List(lst.map(lambda im: ee.List([ee.Image(im).get("system:time_start"),
                                           ee.Image(im).get("system:time_end"),
                                           ee.Image(im).get("sensor")]))).getInfo()
fmt = lambda t: time.strftime("%Y-%m-%d", time.gmtime(t / 1000)) if t else None
dates = [fmt(m[0]) for m in meta]
date_ends = [fmt(m[1]) for m in meta]          # exclusive end from source metadata
sensors = [m[2] for m in meta]
if len(set(zip(dates, sensors))) != n or dates != sorted(dates):
    raise SystemExit("Duplicate or unsorted composites — check the collection.")
print(f"{n} composites: {dates[0]} … {dates[-1]}")
period_median = coll.median()

stack = np.zeros((n, H, W), dtype=np.uint8)
done = [0]


def work(i):
    v = fetch(C.display_filled(lst, n, i, period_median))
    valid = mask & (v >= C.NDVI_RANGE[0] - 1e-6) & (v <= C.NDVI_RANGE[1] + 1e-6)
    if not valid.any():
        raise SystemExit(f"Frame {dates[i]} has no valid cells.")
    q = np.zeros((H, W), dtype=np.uint8)
    q[valid] = 1 + np.round(np.clip(v[valid], 0, NMAX) / NMAX * 254).astype(np.uint8)
    stack[i] = q
    done[0] += 1
    print(f"  {done[0]}/{n}  {dates[i]} ({sensors[i]})")


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(work, range(n)))

payload = {
    "w": W, "h": H, "nmax": NMAX, "bounds": [x0, y0, x1, y1],
    "dates": dates, "date_ends": date_ends, "sensors": sensors,
    "cadence": "monthly" if C.SOURCE == "S2" else "16day", "source": C.SOURCE,
    "display_note": C.DISPLAY_FILL, "qa_policy": C.QA_POLICY, "boundary_asset": C.BOUNDARY_ASSET,
    "dem_id": C.DEM_ID, "dem_max": dem_max, "pipeline_version": C.PIPELINE_VERSION,
    "dem": base64.b64encode(np.round(dem).astype("<u2").tobytes()).decode("ascii"),
    "data": base64.b64encode(stack.tobytes()).decode("ascii"),
}
tmp = OUT + ".tmp"
with open(tmp, "w") as fh:
    json.dump(payload, fh)
os.replace(tmp, OUT)
print(f"\nSaved {OUT}. Next: python build_map.py")
