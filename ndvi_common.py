"""
ndvi_common.py — ONE place for settings and Earth Engine collection logic.
Used by fetch_and_pack.py (display frames) and build_map.py (statistics + HTML).
"""
import ee

PROJECT = "qatar-reef-2026"
BOUNDARY_ASSET = "projects/qatar-reef-2026/assets/pakistan_official"
SOURCE = "MODIS"                          # "MODIS" or "S2"
START, END = "2025-09-01", "2026-10-01"   # composites whose START date is in [START, END)
PROVINCE_ASSET = None          # optional: your own province polygons asset, e.g. "projects/qatar-reef-2026/assets/pak_provinces"
PROVINCE_NAME_FIELD = "name"   # attribute holding the province name in that asset
DEM_ID = "COPERNICUS/DEM/GLO30_2024_1"    # Copernicus surface model (DSM), pinned release
NDVI_RANGE = (-0.2, 1.0)                  # documented valid MODIS NDVI range
CLASSES = [("lower", 0.2, 0.4), ("medium", 0.4, 0.6), ("higher", 0.6, 1.0)]   # last upper bound inclusive
HEADLINE_MIN = 0.4   # headline area counts classes starting at this NDVI (0.2-0.4 = sparse grass/shrubs, shown separately)
QA_POLICY = ("MODIS: SummaryQA <= 1 (good + marginal), NDVI within [-0.2, 1]"
             if SOURCE == "MODIS" else "Sentinel-2: Cloud Score+ cs_cdf >= 0.6")
DISPLAY_FILL = "display frames only: gaps filled with neighbouring-composite mean, then period median"
PIPELINE_VERSION = "2026-10-05b"

if SOURCE not in ("MODIS", "S2"):
    raise SystemExit(f'SOURCE must be "MODIS" or "S2", not {SOURCE!r}')


def init():
    ee.Initialize(project=PROJECT)


def boundary():
    return ee.FeatureCollection(BOUNDARY_ASSET)


def bounds(fc):
    ring = fc.geometry().bounds(maxError=1000).coordinates().getInfo()[0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _modis():
    def prep(tag):
        def _f(img):
            ndvi = img.select("NDVI").multiply(0.0001)
            ok = (img.select("SummaryQA").lte(1)
                  .And(ndvi.gte(NDVI_RANGE[0])).And(ndvi.lte(NDVI_RANGE[1])))
            return ee.Image(ndvi.updateMask(ok).rename("NDVI")
                            .copyProperties(img, ["system:time_start", "system:time_end"])
                            .set("sensor", tag))
        return _f
    t = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(START, END).map(prep("T"))
    a = ee.ImageCollection("MODIS/061/MYD13Q1").filterDate(START, END).map(prep("A"))
    return t.merge(a).sort("system:time_start")


def _s2(region):
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    def s2_ndvi(img):   # keep the timestamp: normalizedDifference() drops image properties
        return ee.Image(img.normalizedDifference(["B8", "B4"]).rename("NDVI")
                        .updateMask(img.select("cs_cdf").gte(0.6))
                        .copyProperties(img, ["system:time_start"]))

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
          .filterDate(START, END).linkCollection(cs, ["cs_cdf"]).map(s2_ndvi))
    empty = ee.Image.constant(0).toFloat().rename("NDVI").updateMask(ee.Image.constant(0))
    sy, sm, ey, em = int(START[:4]), int(START[5:7]), int(END[:4]), int(END[5:7])
    imgs = []
    for k in range((ey - sy) * 12 + (em - sm)):
        d = ee.Date(START).advance(k, "month")
        e = d.advance(1, "month")
        month = s2.filterDate(d, e)
        comp = ee.Image(ee.Algorithms.If(month.size().gt(0), month.median(), empty))
        imgs.append(comp.set({"system:time_start": d.millis(), "system:time_end": e.millis(),
                              "sensor": "S", "scene_count": month.size()}))
    return ee.ImageCollection(imgs)


def ndvi_collection(region):
    """QA-masked NDVI composites (observed values only, no filling)."""
    return _modis() if SOURCE == "MODIS" else _s2(region)


def display_filled(lst, n, i, period_median):
    """Gap-filled frame for the ANIMATION ONLY. Never used for statistics."""
    img = ee.Image(lst.get(i))
    nbrs = [ee.Image(lst.get(j)) for j in (i - 1, i + 1) if 0 <= j < n]
    if nbrs:
        img = img.unmask(ee.ImageCollection(nbrs).mean())
    return img.unmask(period_median)