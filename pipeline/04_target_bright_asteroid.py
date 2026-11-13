# ============================================================
# SPHEREx Planet X - Step 4: "target mode" - recover a BRIGHT known asteroid.
# Each full SPHEREx image is ~3.5 x 3.5 degrees, much bigger than our cutout.
# So: ask JPL which known asteroids brighter than V=17 were anywhere inside
# images A and B, pick the brightest, and cut out a small box around it.
# Run AFTER Steps 2 and 3 (reuses res, ia, ib, mjd, fetch_cutout, sexa, API, noise).
# ============================================================
import warnings
import requests
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from reproject import reproject_interp
warnings.filterwarnings("ignore")

V_BRIGHT, V_FAINT = 11.0, 17.5   # brighter than 11 may saturate; fainter than 17 hard to see
BOX = 0.12                       # deg, size of the zoomed cutout around the asteroid
HALF = 1.5                       # deg, half-width of a full SPHEREx image


def sb_query(mjd_time, ra, dec, hw, vlim):
    """Known small bodies within +-hw deg of (ra, dec) at mjd_time, brighter than vlim."""
    params = {"obs-time": f"{Time(mjd_time, format='mjd').jd:.6f}",
              "fov-ra-center": sexa(ra % 360, True), "fov-dec-center": sexa(dec, False),
              "fov-ra-hwidth": hw, "fov-dec-hwidth": hw,
              "vmag-lim": vlim, "two-pass": "false"}
    js = {}
    for attempt in range(3):
        for where in [{"xobs": "0.0,0.0,0.0"}, {"lat": 0, "lon": 0, "alt": 0}]:
            try:
                js = requests.get(API, params={**params, **where}, timeout=180).json()
            except Exception as e:
                print(f"  JPL slow ({type(e).__name__}), retrying...")
                js = {}
                continue
            if "fields_first" in js:
                break
        if "fields_first" in js:
            break
    if "fields_first" not in js:
        print("JPL error:", js)
        return {}
    f = js["fields_first"]
    i_ra = [i for i, x in enumerate(f) if x.startswith("Astrometric RA")][0]
    i_de = [i for i, x in enumerate(f) if x.startswith("Astrometric Dec")][0]
    i_v = [i for i, x in enumerate(f) if x.startswith("Visual magnitude")][0]
    out = {}
    for row in js.get("data_first_pass", []):
        c = SkyCoord(str(row[i_ra]), str(row[i_de]).replace("'", " ").replace('"', " "),
                     unit=(u.hourangle, u.deg))
        try:
            v = float(row[i_v])
        except (TypeError, ValueError):
            continue
        out[row[0]] = (c.ra.deg, c.dec.deg, v)
    return out


# 1. Centre of each full image, then ask JPL what bright objects were inside
cA = (float(res[ia]["s_ra"]), float(res[ia]["s_dec"]))
cB = (float(res[ib]["s_ra"]), float(res[ib]["s_dec"]))
print("Asking JPL about the full images (this can take a minute)...")
inA = sb_query(mjd[ia], *cA, HALF, V_FAINT)
inB = sb_query(mjd[ib], *cB, HALF, V_FAINT)
both = {n: (inA[n], inB[n]) for n in inA if n in inB and inA[n][2] >= V_BRIGHT}
print(f"{len(inA)} objects in A, {len(inB)} in B, {len(both)} usable in both (V {V_BRIGHT}-{V_FAINT})")
if not both:
    raise SystemExit("No bright known asteroid in this pair - try a different RA/DEC in cell 2.")

for n, (a, b) in sorted(both.items(), key=lambda kv: kv[1][0][2])[:8]:
    print(f"  {str(n):<30} V={a[2]:5.1f}")

# 2. Try candidates from brightest down. Each image is cut out around the
#    asteroid's OWN position at that time (so it is surely inside that image).
def load_at(i, ra, dec, size):
    h = fetch_cutout(res[i]["access_url"], ra, dec, size, retries=1)
    if h is None:
        return None, None
    img = h["IMAGE"].data.astype(float) - h["ZODI"].data
    img[(h["FLAGS"].data & BAD_BITS) != 0] = np.nan
    img -= np.nanmedian(img)
    return img, WCS(h["IMAGE"].header)


TA = None
for name, (pa, pb) in sorted(both.items(), key=lambda kv: kv[1][0][2]):
    move = SkyCoord(pa[0], pa[1], unit="deg").separation(
        SkyCoord(pb[0], pb[1], unit="deg")).arcsec
    print(f"\nTrying {name}  V={pa[2]:.1f}  moved {move:.0f}\" (~{move/6.15:.0f} px) in "
          f"{(mjd[ib]-mjd[ia])*24:.1f} h")
    size = max(BOX, 3 * move / 3600)
    ra_c, dec_c = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
    TA, tA = load_at(ia, ra_c, dec_c, size)
    if TA is None:                       # midpoint off the edge -> centre on own position
        TA, tA = load_at(ia, pa[0], pa[1], size)
    TB, tB = load_at(ib, ra_c, dec_c, size)
    if TB is None:
        TB, tB = load_at(ib, pb[0], pb[1], size)
    if TA is not None and TB is not None:
        break
    print("  this one is too close to an image edge, trying the next...")
    TA = None

if TA is None:
    raise SystemExit("None of the bright asteroids could be cut out - try another RA/DEC in cell 2.")

TB_al, _ = reproject_interp((TB, tB), tA, shape_out=TA.shape)
m = (TA > 10 * noise(TA)) & (TB_al > 10 * noise(TA))
k = np.nanmedian(TA[m] / TB_al[m]) if m.sum() > 5 else 1.0
TD = TA - k * TB_al
sD = noise(TD)

xa, ya = tA.world_to_pixel(SkyCoord(pa[0], pa[1], unit="deg"))
xb, yb = tA.world_to_pixel(SkyCoord(pb[0], pb[1], unit="deg"))

# 3. Measure: is there a bright (+) spot where JPL says it was at A, and (-) at B?
def peak_snr(img, x, y, r=2):
    x, y = int(round(float(x))), int(round(float(y)))
    patch = img[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
    return np.nanmax(np.abs(patch)) / sD if patch.size else np.nan

snrA, snrB = peak_snr(TD, xa, ya), peak_snr(TD, xb, yb)
print(f"Signal at predicted A position: {snrA:.1f} sigma   at B position: {snrB:.1f} sigma")
print("DETECTED!" if min(snrA, snrB) >= 5 else "Not clearly detected (below 5 sigma).")

# 4. Picture for the demo
fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
for a, im, t in zip(ax[:2], [TA, k * TB_al],
                    [f"A  {Time(mjd[ia], format='mjd').iso[:16]} UTC",
                     f"B  {Time(mjd[ib], format='mjd').iso[:16]} UTC"]):
    lo, hi = np.nanpercentile(im, [5, 99.7])
    a.imshow(im, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    a.set_title(t)
ax[2].imshow(TD, origin="lower", cmap="RdBu_r", vmin=-6 * sD, vmax=6 * sD)
ax[2].set_title(f"A - B   ({name}, V={pa[2]:.1f})")
for a in ax:
    a.plot(xa, ya, "o", mfc="none", mec="gold", ms=18, mew=2)
    a.plot(xb, yb, "s", mfc="none", mec="cyan", ms=16, mew=2)
    a.annotate("", xy=(xb, yb), xytext=(xa, ya),
               arrowprops=dict(arrowstyle="->", color="gold", lw=1.5))
    a.axis("off")
fig.suptitle(f"SPHEREx recovers known asteroid {name}: gold = JPL position at A, "
             f"cyan = at B ({(mjd[ib]-mjd[ia])*24:.1f} h later)", fontsize=13)
plt.tight_layout()
plt.savefig("step4_target.png", dpi=130)
plt.show()
