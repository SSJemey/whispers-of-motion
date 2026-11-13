"""
SPHEREx Planet X - Cell 2: settings and helper functions.
Run this cell before any other SPHEREx cell. It only defines things,
so it finishes instantly. (To download the 12 preview cutouts again,
change the last line to main().)
"""

import os
import time
import http.client
import urllib.error

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import conf
from astroquery.ipac.irsa import Irsa

conf.remote_timeout = 120  # SPHEREx files are slow to serve

# ---------------- settings ----------------
RA, DEC = 0.0, 0.0          # deg. On the ecliptic -> lots of asteroids. Change freely.
COLLECTION = "spherex_qr2"  # QR3 exists but its search index is still empty
CUTOUT_DEG = 0.2            # cutout side in degrees (~117 SPHEREx pixels of 6.15")
MAX_CUTOUTS = 12            # keep small while testing
DETECTOR = None             # e.g. 1..6 to compare like-with-like wavelengths; None = all
OUT_DIR = "cutouts"
# ------------------------------------------


def _query_sia(ra, dec):
    """Search method 1: IRSA's Simple Image Access (SIA) service."""
    coord = SkyCoord(ra, dec, unit="deg")
    res = Irsa.query_sia(pos=(coord, 1 * u.arcsec), collection=COLLECTION)
    return res


def _query_tap(ra, dec):
    """Search method 2 (backup): IRSA's TAP database, same as IRSA's cutout tutorial."""
    q = f"""
    SELECT 'https://irsa.ipac.caltech.edu/' || a.uri AS access_url,
           p.time_bounds_lower AS t_min
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
    """
    res = Irsa.query_tap(q).to_table()
    keep = [("level2" in str(x)) and str(x).endswith(".fits") for x in res["access_url"]]
    return res[keep]


def query_images(ra, dec, tries=3):
    """Ask IRSA which SPHEREx images cover (ra, dec).
    Tries SIA first; if it returns nothing, falls back to TAP. Retries both."""
    res = None
    for attempt in range(tries):
        for name, fn in [("SIA", _query_sia), ("TAP", _query_tap)]:
            try:
                res = fn(ra, dec)
                if len(res) > 0:
                    print(f"{len(res)} images cover RA={ra}, Dec={dec} (via {name})")
                    res.sort("t_min")  # MJD of exposure start
                    return res
                print(f"  {name} returned 0 images")
            except Exception as e:
                print(f"  {name} error: {str(e)[:120]}")
        print(f"Retrying in 20 s... ({attempt + 1}/{tries})")
        time.sleep(20)
    raise RuntimeError("IRSA search not responding - wait a few minutes and try again.")


def detector_of(row):
    # Detector number is in the filename: ..._<obs>D<detector>_spx_...
    name = str(row["access_url"]).split("/")[-1]
    return int(name.split("_spx")[0].split("D")[-1])


def fetch_cutout(url, ra, dec, size_deg, retries=3):
    cut_url = f"{url}?center={ra},{dec}d&size={size_deg}"
    for attempt in range(retries):
        try:
            with fits.open(cut_url, memmap=False) as hdul:
                for h in hdul:
                    _ = h.data  # force load before the file closes
                return fits.HDUList([h.copy() for h in hdul])
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError,
                http.client.IncompleteRead, OSError) as e:
            if attempt == retries - 1:
                print("  failed:", e)
                return None
            time.sleep(10 * (attempt + 1))


def central_wavelength(hdul):
    """Approximate wavelength (um) at the cutout centre, from the spectral WCS 'W'."""
    try:
        sw = WCS(header=hdul["IMAGE"].header, fobj=hdul, key="W")
        sw.sip = None
        ny, nx = hdul["IMAGE"].data.shape
        wl, _ = sw.pixel_to_world(nx / 2, ny / 2)
        return float(wl.to_value(u.um))
    except Exception:
        return np.nan


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = query_images(RA, DEC)

    rows = [r for r in res if DETECTOR is None or detector_of(r) == DETECTOR]
    rows = rows[:MAX_CUTOUTS]

    saved = []
    print(f"\n{'#':>3} {'UTC time':<20} {'gap(h)':>8} {'det':>4} {'lambda(um)':>11}")
    prev_mjd = None
    for i, row in enumerate(rows):
        mjd = float(row["t_min"])
        hdul = fetch_cutout(row["access_url"], RA, DEC, CUTOUT_DEG)
        if hdul is None:
            continue
        wl = central_wavelength(hdul)
        gap = "" if prev_mjd is None else f"{(mjd - prev_mjd) * 24:8.2f}"
        prev_mjd = mjd
        t = Time(mjd, format="mjd").iso[:19]
        det = detector_of(row)
        print(f"{i:>3} {t:<20} {gap:>8} {det:>4} {wl:>11.3f}")

        fname = os.path.join(OUT_DIR, f"cut_{i:03d}_mjd{mjd:.4f}_D{det}.fits")
        hdul.writeto(fname, overwrite=True)
        saved.append((fname, t, det, wl))

    if not saved:
        print("No cutouts downloaded.")
        return

    n = len(saved)
    cols = min(4, n)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (fname, t, det, wl) in zip(axes.ravel(), saved):
        img = fits.getdata(fname, "IMAGE")
        lo, hi = np.nanpercentile(img, [5, 99.5])
        ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"{t}\nD{det}  {wl:.2f} um", fontsize=9)
    plt.tight_layout()
    plt.savefig("step1_epochs.png", dpi=120)
    plt.show()
    print(f"\nSaved {n} cutouts to ./{OUT_DIR}/ and a preview to step1_epochs.png")


print("Cell 2 loaded: settings and functions ready.")
# main()   # <- remove the # to download the 12 preview cutouts again
