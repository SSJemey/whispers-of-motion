!pip install -q reproject
# ============================================================
# SPHEREx Planet X - Step 2: pick two visits a few hours apart,
# align them, subtract, and look for things that moved.
# Run AFTER the main script cell (it reuses query_images,
# detector_of, fetch_cutout, RA, DEC).
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.time import Time
from reproject import reproject_interp
from scipy import ndimage

CUT = 0.3                         # deg; bigger than before so movers stay in frame
MIN_H, MAX_H, TARGET_H = 1, 12, 4  # want a pair ~4 h apart (1-12 h allowed)

# ---------- 1. choose the best pair (same detector, ~4 h apart) ----------
res = query_images(RA, DEC)
dets = np.array([detector_of(r) for r in res])
mjd = np.array(res["t_min"], dtype=float)

best = None
for d in sorted(set(dets)):
    idx = np.where(dets == d)[0]
    idx = idx[np.argsort(mjd[idx])]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            gap = (mjd[idx[b]] - mjd[idx[a]]) * 24
            if gap > MAX_H:
                break
            if gap >= MIN_H:
                score = abs(gap - TARGET_H)
                if best is None or score < best[0]:
                    best = (score, d, int(idx[a]), int(idx[b]), gap)

_, DET, ia, ib, gap = best
print(f"Chosen pair: detector D{DET}, {gap:.2f} hours apart")
print("  A:", Time(mjd[ia], format="mjd").iso[:19])
print("  B:", Time(mjd[ib], format="mjd").iso[:19])

# ---------- 2. download both cutouts, remove background, mask bad pixels ----------
BAD_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 6) | (1 << 10) | (1 << 11)
# transient(cosmic ray), overflow, onboard error, unusable, hot, cold

def load(i):
    h = fetch_cutout(res[i]["access_url"], RA, DEC, CUT)
    img = h["IMAGE"].data.astype(float) - h["ZODI"].data   # remove zodiacal glow
    img[(h["FLAGS"].data & BAD_BITS) != 0] = np.nan          # hide bad pixels
    img -= np.nanmedian(img)                                 # remaining sky level -> 0
    return img, WCS(h["IMAGE"].header)

A, wA = load(ia)
B, wB = load(ib)

# ---------- 3. align B onto A's pixel grid ----------
B_al, _ = reproject_interp((B, wB), wA, shape_out=A.shape)

# Each visit sees a slightly different wavelength, so stars change brightness.
# Scale B so bright stars match A on average.
def noise(x):
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))

sA = noise(A)
m = (A > 10 * sA) & (B_al > 10 * sA)
k = np.nanmedian(A[m] / B_al[m]) if m.sum() > 20 else 1.0
diff = A - k * B_al
sD = noise(diff)
print(f"Brightness scale B->A: {k:.3f}   noise in difference: {sD:.4f} MJy/sr")

# ---------- 4. find blobs in the difference image ----------
# Something that MOVED leaves a pair: + where it was in A, - where it was in B.
found = {}
for sign, name in [(1, "in A only (+)"), (-1, "in B only (-)")]:
    mask = np.nan_to_num(sign * diff) > 5 * sD
    lab, n = ndimage.label(mask)
    if n == 0:
        found[name] = []
        continue
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    cents = ndimage.center_of_mass(mask, lab, range(1, n + 1))
    found[name] = [c for c, s in zip(cents, sizes) if s >= 2]   # ignore single pixels
    print(f"{len(found[name]):3d} blobs {name}")

# ---------- 5. show A, B, and A - B ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
for a, im, title in zip(ax[:2], [A, k * B_al],
                        [f"A  {Time(mjd[ia], format='mjd').iso[:16]}",
                         f"B (aligned)  {Time(mjd[ib], format='mjd').iso[:16]}"]):
    lo, hi = np.nanpercentile(im, [5, 99.5])
    a.imshow(im, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    a.set_title(title)
ax[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-5 * sD, vmax=5 * sD)
ax[2].set_title(f"A - B  (red = only in A, blue = only in B), gap {gap:.1f} h")
for (name, pts), col in zip(found.items(), ["lime", "orange"]):
    for y, x in pts:
        ax[2].add_patch(plt.Circle((x, y), 4, fill=False, color=col, lw=1.2))
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig("step2_difference.png", dpi=120)
plt.show()
