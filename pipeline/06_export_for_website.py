# ============================================================
# SPHEREx Planet X - Export for the website.
# Saves the images (PNG) and all numbers (data.json) into a folder "web",
# then zips it as spherex_web.zip (download it from the Output panel).
# Run AFTER Step 5.
# ============================================================
import os, json, shutil
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

os.makedirs("web", exist_ok=True)

def save_gray(img, path, lo=None, hi=None):
    if lo is None:
        lo, hi = np.nanpercentile(img, [5, 99.5])
    plt.imsave(path, np.flipud(np.nan_to_num(img, nan=lo)), cmap="gray", vmin=lo, vmax=hi)

def save_diff(img, s, path):
    plt.imsave(path, np.flipud(np.nan_to_num(img)), cmap="RdBu_r", vmin=-6 * s, vmax=6 * s)

# ---- wide field (Step 2/3): the whole 0.3 deg patch ----
sW = noise(diff)
lo, hi = np.nanpercentile(A, [5, 99.5])
save_gray(A, "web/wide_A.png", lo, hi)
save_gray(A - diff, "web/wide_B.png", lo, hi)        # = k * B aligned
save_diff(diff, sW, "web/wide_D.png")
nyW, nxW = A.shape
known = []
for nm in sorted(set(pA) | set(pB)):
    item = {"name": str(nm), "v": float((pA.get(nm) or pB.get(nm))[2])}
    if nm in pA:
        item["xa"], item["ya"] = pA[nm][0] / nxW, 1 - pA[nm][1] / nyW
    if nm in pB:
        item["xb"], item["yb"] = pB[nm][0] / nxW, 1 - pB[nm][1] / nyW
    known.append(item)

# ---- zoom on the confirmed asteroid (Step 5) ----
zA, zB, zD = TA[sl], (k * TB_al)[sl], TD[sl]
lo, hi = np.nanpercentile(zA, [5, 99.5])
save_gray(zA, "web/zoom_A.png", lo, hi)
save_gray(zB, "web/zoom_B.png", lo, hi)
save_diff(zD, sD, "web/zoom_D.png")
nyZ, nxZ = zA.shape
nrm = lambda x, y: ((x - off[0]) / nxZ, 1 - (y - off[1]) / nyZ)
mA, mB = nrm(x1, y1), nrm(x2, y2)
jA, jB = nrm(float(xa), float(ya)), nrm(float(xb), float(yb))

data = {
    "collection": COLLECTION,
    "detector": int(DET),
    "field": {"ra": RA, "dec": DEC, "size_deg": CUT},
    "timeA": Time(mjd[ia], format="mjd").iso[:19] + " UTC",
    "timeB": Time(mjd[ib], format="mjd").iso[:19] + " UTC",
    "gap_h": round(float((mjd[ib] - mjd[ia]) * 24), 2),
    "n_images": int(len(res)),
    "wide": {"A": "wide_A.png", "B": "wide_B.png", "D": "wide_D.png",
             "w": nxW, "h": nyW, "known": known},
    "target": {
        "name": str(name), "v": float(pa[2]),
        "A": "zoom_A.png", "B": "zoom_B.png", "D": "zoom_D.png", "w": nxZ, "h": nyZ,
        "measA": mA, "measB": mB, "jplA": jA, "jplB": jB,
        "sigmaA": round(s1, 1), "sigmaB": round(s2, 1),
        "motion_pred_px": round(float(pred_motion), 1),
        "motion_meas_px": round(float(seen_motion), 1),
        "motion_arcsec": round(float(seen_motion) * 6.15, 0),
        "offset_agree_px": round(float(mismatch), 1),
    },
}
with open("web/data.json", "w") as f:
    json.dump(data, f, indent=2)

for p in ["step2_difference.png", "step3_known_asteroids.png", "step5_asteroid_zoom.png"]:
    if os.path.exists(p):
        shutil.copy(p, "web/" + p)

shutil.make_archive("spherex_web", "zip", "web")
print("Saved:", sorted(os.listdir("web")))
print("\nDownload /kaggle/working/spherex_web.zip from the Output panel on the right.")
