# ============================================================
# SPHEREx Planet X - Step 3: which KNOWN asteroids should be here?
# Asks NASA JPL's Small-Body Identification service which catalogued
# asteroids/comets were in our patch of sky at time A and time B,
# then marks them on the difference image and checks if we detected them.
# Run AFTER Step 2 (it reuses A, diff, sD, wA, found, mjd, ia, ib, RA, DEC, CUT).
# ============================================================
import requests
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

API = "https://ssd-api.jpl.nasa.gov/sb_ident.api"


def sexa(value, hours):
    """Degrees -> 'hh-mm-ss.ss' (RA) or 'dd-mm-ss.s' with M for minus (Dec)."""
    v = value / 15.0 if hours else value
    sign = "M" if v < 0 else ""
    v = abs(v)
    d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
    return f"{sign}{d:02d}-{m:02d}-{s:05.2f}"


def known_objects(mjd_time):
    """List of (name, ra_deg, dec_deg, V mag) of known small bodies in the field."""
    base = {
        "obs-time": f"{Time(mjd_time, format='mjd').jd:.6f}",
        "fov-ra-center": sexa(RA % 360, hours=True),
        "fov-dec-center": sexa(DEC, hours=False),
        "fov-ra-hwidth": CUT / 2 + 0.05,
        "fov-dec-hwidth": CUT / 2 + 0.05,
        "two-pass": "false",   # fast mode (the precise 2nd pass can take minutes)
    }
    # Observer: Earth's centre (SPHEREx orbits only ~700 km up, so this is close enough)
    js = {}
    for attempt in range(3):
        for where in [{"xobs": "0.0,0.0,0.0"}, {"lat": 0, "lon": 0, "alt": 0}]:
            try:
                r = requests.get(API, params={**base, **where}, timeout=180)
                js = r.json()
            except Exception as e:
                print(f"  JPL slow/unreachable ({type(e).__name__}), retrying...")
                js = {}
                continue
            if "fields_second" in js or "fields_first" in js:
                break
        if "fields_second" in js or "fields_first" in js:
            break
    else:
        print("JPL error:", js)
        return []

    fields = js.get("fields_second") or js.get("fields_first")
    rows = js.get("data_second_pass") or js.get("data_first_pass") or []
    i_name = fields.index("Object name")
    i_ra = [i for i, f in enumerate(fields) if f.startswith("Astrometric RA")][0]
    i_dec = [i for i, f in enumerate(fields) if f.startswith("Astrometric Dec")][0]
    i_v = [i for i, f in enumerate(fields) if f.startswith("Visual magnitude")][0]

    out = []
    for row in rows:
        dec_txt = str(row[i_dec]).replace("'", " ").replace('"', " ")
        c = SkyCoord(str(row[i_ra]), dec_txt, unit=(u.hourangle, u.deg))
        try:
            vmag = float(row[i_v])
        except (TypeError, ValueError):
            vmag = np.nan
        out.append((row[i_name], c.ra.deg, c.dec.deg, vmag))
    return out


objA = known_objects(mjd[ia])
objB = known_objects(mjd[ib])
print(f"Known objects in field: {len(objA)} at time A, {len(objB)} at time B")

# Convert sky positions to pixels on A's grid (B was aligned onto it)
ny, nx = A.shape

def to_pix(objs):
    res = {}
    for name, ra, dec, v in objs:
        x, y = wA.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
        if 0 <= x < nx and 0 <= y < ny:
            res[name] = (float(x), float(y), v)
    return res

pA, pB = to_pix(objA), to_pix(objB)

# Did our blob finder catch them? (within 3 pixels = ~18 arcsec)
plus = np.array(found.get("in A only (+)", []))   # (y, x)
minus = np.array(found.get("in B only (-)", []))

def hit(blobs, x, y, r=3):
    if len(blobs) == 0:
        return False
    return bool(np.min(np.hypot(blobs[:, 1] - x, blobs[:, 0] - y)) <= r)

print(f"\n{'object':<28}{'V mag':>6}  {'seen at A':>9}  {'seen at B':>9}")
n_found = 0
for name in sorted(set(pA) | set(pB)):
    v = (pA.get(name) or pB.get(name))[2]
    a = hit(plus, *pA[name][:2]) if name in pA else None
    b = hit(minus, *pB[name][:2]) if name in pB else None
    n_found += bool(a) or bool(b)
    fmt = lambda s: "-" if s is None else ("YES" if s else "no")
    print(f"{str(name):<28}{v:6.1f}  {fmt(a):>9}  {fmt(b):>9}")
print(f"\nDetected {n_found} of {len(set(pA) | set(pB))} known objects in the image.")

# Plot: difference image with known objects marked
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-5 * sD, vmax=5 * sD)
for name, (x, y, v) in pA.items():
    ax.plot(x, y, "o", mfc="none", mec="gold", ms=16, mew=2)
    if name in pB:
        xb, yb, _ = pB[name]
        ax.annotate("", xy=(xb, yb), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="gold", lw=1.5))
    ax.text(x + 4, y + 4, str(name), color="black", fontsize=8,
            bbox=dict(fc="gold", alpha=0.7, lw=0, pad=1))
for name, (x, y, v) in pB.items():
    ax.plot(x, y, "s", mfc="none", mec="cyan", ms=14, mew=2)
ax.set_title(f"Known asteroids (JPL): gold circle = position at A, cyan square = at B\n"
             f"red = only in A, blue = only in B,  gap {(mjd[ib]-mjd[ia])*24:.1f} h")
ax.axis("off")
plt.tight_layout()
plt.savefig("step3_known_asteroids.png", dpi=120)
plt.show()
