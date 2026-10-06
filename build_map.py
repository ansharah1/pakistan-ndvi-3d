"""
build_map.py — statistics on OBSERVED pixels + final offline HTML (pakistan_ndvi_map.html).
Needs: pak_ndvi_frames.json (from fetch_and_pack.py), viewer_template.html, ndvi_common.py
Statistics: QA-masked NDVI only (no gap filling), on ONE fixed analysis grid for the whole series.
"""
import hashlib, json, os, time, urllib.request, warnings
from concurrent.futures import ThreadPoolExecutor
import ee
import ndvi_common as C

warnings.filterwarnings("ignore", category=DeprecationWarning)
WORKERS = 6
DATA, TEMPLATE, OUT = "pak_ndvi_frames.json", "viewer_template.html", "pakistan_ndvi_map.html"
FALLBACK_RES = 0.00225     # ~250 m EPSG:4326 grid, used for the WHOLE series only if the native grid fails
LIBS = ["https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
        "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"]

with open(DATA) as fh:
    D = json.load(fh)
n = len(D["dates"])
for key in ("w", "h", "bounds", "dates", "sensors", "data", "source"):
    if key not in D:
        raise SystemExit(f"{DATA} is missing '{key}'. Re-run fetch_and_pack.py.")
if D["source"] != C.SOURCE or D.get("pipeline_version") != C.PIPELINE_VERSION:
    raise SystemExit(f"{DATA} was made with different settings/version. Re-run fetch_and_pack.py.")

C.init()
asset_time = ee.data.getAsset(C.BOUNDARY_ASSET).get("updateTime", "")
fingerprint = hashlib.sha256(json.dumps({
    "boundary": C.BOUNDARY_ASSET, "boundary_time": asset_time, "dates": D["dates"], "sensors": D["sensors"],
    "source": C.SOURCE, "qa": C.QA_POLICY, "range": C.NDVI_RANGE, "classes": C.CLASSES,
    "start": C.START, "end": C.END, "version": C.PIPELINE_VERSION}, sort_keys=True).encode()).hexdigest()[:16]

if D.get("stats_fingerprint") == fingerprint and len(D.get("stats", [])) == n:
    print("Statistics up to date (fingerprint match) — skipping Earth Engine.")
else:
    pak = C.boundary()
    x0, y0, x1, y1 = D["bounds"]
    region = ee.Geometry.Rectangle([x0, y0, x1, y1], "EPSG:4326", False)
    bmask = ee.Image(1).clipToCollection(pak)
    km2 = ee.Image.pixelArea().divide(1e6)
    coll = C.ndvi_collection(region)
    dates = [time.strftime("%Y-%m-%d", time.gmtime(t / 1000))
             for t in coll.aggregate_array("system:time_start").getInfo()]
    if dates != D["dates"]:
        raise SystemExit("Composite dates differ from the JSON. Re-run fetch_and_pack.py first.")
    lst = coll.toList(n)

    if C.SOURCE == "MODIS":
        grid = {"crs": ee.Image(lst.get(0)).select("NDVI").projection()}
        grid_label = "MODIS native sinusoidal grid (~232 m pixels)"
    else:
        grid = None
    fallback = {"crs": "EPSG:4326", "crsTransform": [FALLBACK_RES, 0, x0, 0, -FALLBACK_RES, y1]}
    fallback_label = f"fixed EPSG:4326 grid, {FALLBACK_RES}° (~250 m)"

    def frame_image(i):
        raw = ee.Image(lst.get(i)).select("NDVI")          # QA-masked, NOT filled
        bands = [km2.updateMask(raw.mask().gt(0)).rename("observed"),
                 raw.multiply(km2).rename("ndvi_area")]
        for k, (name, lo, hi) in enumerate(C.CLASSES):
            upper = raw.lte(hi) if k == len(C.CLASSES) - 1 else raw.lt(hi)
            bands.append(km2.updateMask(raw.gte(lo).And(upper)).rename(name))
        return ee.Image.cat(bands).updateMask(bmask)

    def reduce(img, g, tries=3):
        for attempt in range(tries):
            try:
                return img.reduceRegion(ee.Reducer.sum(), region, maxPixels=1e11, tileScale=8, **g).getInfo()
            except Exception as e:
                if attempt == tries - 1:
                    raise
                print(f"  retry {attempt + 1}: {str(e)[:90]}")
                time.sleep(6 * (attempt + 1))

    # choose ONE grid for the whole series: test the native grid on the first frame
    if grid is not None:
        try:
            reduce(frame_image(0), grid, tries=2)
        except Exception as e:
            print(f"Native grid failed ({str(e)[:80]}); using {fallback_label} for ALL frames.")
            grid, grid_label = None, None
    if grid is None:
        grid, grid_label = fallback, fallback_label
    print("Analysis grid:", grid_label)

    total = reduce(km2.updateMask(bmask).rename("t"), grid)["t"]
    print(f"Mapped boundary area: {total:,.0f} km²  (check this matches the boundary you intend)")

    stats = [None] * n

    def work(i):
        r = reduce(frame_image(i), grid)
        obs = r.get("observed") or 0
        rec = {"observed": round(obs)}
        for name, _, _ in C.CLASSES:
            rec[name] = round(r.get(name) or 0)
        rec["mean_ndvi"] = round((r.get("ndvi_area") or 0) / obs, 4) if obs else None
        stats[i] = rec
        g = sum(rec[c] for c, _, _ in C.CLASSES)
        print(f"  {D['dates'][i]}  observed {obs / total * 100:5.1f}%  NDVI≥0.2 {g:>9,} km² "
              f"({g / obs * 100 if obs else 0:4.1f}% of observed)  mean NDVI {rec['mean_ndvi']}")

    print(f"Computing statistics for {n} composites (~5–10 min)…")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, range(n)))

    for old in ("area", "area_total", "area_scale_m", "area_classes", "means"):
        D.pop(old, None)
    D["stats"] = stats
    D["stats_meta"] = {"boundary_area_km2": round(total), "grid": grid_label, "qa_policy": C.QA_POLICY,
                       "classes": C.CLASSES, "observed_only": True, "boundary_asset": C.BOUNDARY_ASSET}
    D["stats_fingerprint"] = fingerprint
    tmp = DATA + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(D, fh)
    os.replace(tmp, DATA)
    print("Statistics saved into", DATA)

# ---------------- provinces on the display grid (hover, click-to-zoom, province series) ----------------
prov_key = {"asset": C.PROVINCE_ASSET or "auto", "w": D["w"], "h": D["h"], "bounds": D["bounds"]}
if (D.get("provinces") or {}).get("key") != prov_key:
    def province_fc():
        if C.PROVINCE_ASSET:
            return (ee.FeatureCollection(C.PROVINCE_ASSET), C.PROVINCE_NAME_FIELD,
                    f"your asset {C.PROVINCE_ASSET}")
        try:
            fc = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM1").filter(ee.Filter.eq("shapeGroup", "PAK"))
            if fc.size().getInfo() > 0:
                return fc, "shapeName", "geoBoundaries v6.0.0 ADM1 (gbOpen)"
        except Exception as e:
            print(f"geoBoundaries not available ({str(e)[:60]}), trying FAO GAUL")
        fc = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
        return fc, "ADM1_NAME", "FAO GAUL 2015 level 1"

    fc, field, src = province_fc()
    names = fc.aggregate_array(field).getInfo()
    if not names or len(names) > 250:
        raise SystemExit(f"Province source returned {len(names)} features; check PROVINCE_ASSET / field.")
    flist = fc.toList(len(names))
    fc_id = ee.FeatureCollection(ee.List.sequence(0, len(names) - 1).map(
        lambda j: ee.Feature(flist.get(j)).set("pid", ee.Number(j).add(1))))
    x0, y0, x1, y1 = D["bounds"]
    res = (x1 - x0) / D["w"]
    grid = {"dimensions": {"width": D["w"], "height": D["h"]},
            "affineTransform": {"scaleX": res, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -res, "translateY": y1},
            "crsCode": "EPSG:4326"}
    import base64, numpy as np
    arr = ee.data.computePixels({"expression": ee.Image(0).byte().paint(fc_id, "pid").rename("v"),
                                 "fileFormat": "NUMPY_NDARRAY", "grid": grid})
    pid = np.asarray(arr["v"]).astype(np.uint8)
    D["provinces"] = {"key": prov_key, "names": names, "source": src + ", rasterised to the ~3 km display grid",
                      "grid": base64.b64encode(pid.tobytes()).decode("ascii")}
    print(f"Provinces ({src}): " + ", ".join(names))
    tmp = DATA + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(D, fh)
    os.replace(tmp, DATA)

# ---------------- final HTML ----------------
D["headline_min"] = C.HEADLINE_MIN   # display choice only; no recompute needed
with open(TEMPLATE, encoding="utf-8") as fh:
    html = fh.read()
libs = []
for url in LIBS:
    try:
        libs.append("<script>" + urllib.request.urlopen(url, timeout=30).read().decode("utf-8") + "</script>")
    except Exception as e:
        print(f"WARNING: could not embed {url.split('/')[-1]} ({e}). The map will need internet to open.")
        libs.append(f'<script src="{url}"></script>')
html = html.replace("<!--__LIBS__-->", "\n".join(libs)).replace("/*__NDVI_DATA__*/null", json.dumps(D))
tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    fh.write(html)
os.replace(tmp, OUT)
print(f"\nDone: {OUT} — double-click to open.")