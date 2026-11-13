# ============================================================
# SPHEREx Planet X - Step 5: measure the asteroid properly.
# Looks for the strongest + spot near JPL's A position and the strongest
# - spot near JPL's B position (within 8 pixels), then checks that both
# are shifted by the SAME amount - the signature of a real moving object.
# Run AFTER Step 4 (reuses TD, TA, TB_al, k, sD, xa, ya, xb, yb, name, pa, mjd, ia, ib).
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

R = 8  # search radius in pixels (~50 arcsec)

def strongest(img, x0, y0, sign):
    """Pixel with the largest sign*value within R pixels of (x0, y0)."""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    near = np.hypot(xx - x0, yy - y0) <= R
    vals = np.where(near, np.nan_to_num(sign * img, nan=-np.inf), -np.inf)
    j = np.argmax(vals)
    y, x = np.unravel_index(j, img.shape)
    return float(x), float(y), float(vals[y, x] / sD)

x1, y1, s1 = strongest(TD, float(xa), float(ya), +1)   # bright in A only
x2, y2, s2 = strongest(TD, float(xb), float(yb), -1)   # bright in B only
dA = (x1 - float(xa), y1 - float(ya))
dB = (x2 - float(xb), y2 - float(yb))
mismatch = np.hypot(dA[0] - dB[0], dA[1] - dB[1])
pred_motion = np.hypot(float(xb) - float(xa), float(yb) - float(ya))
seen_motion = np.hypot(x2 - x1, y2 - y1)

print(f"Target: {name}  (V = {pa[2]:.1f})")
print(f"  A: red spot  {s1:5.1f} sigma, offset from JPL ({dA[0]:+.1f}, {dA[1]:+.1f}) px")
print(f"  B: blue spot {s2:5.1f} sigma, offset from JPL ({dB[0]:+.1f}, {dB[1]:+.1f}) px")
print(f"  Motion predicted by JPL: {pred_motion:.1f} px   measured: {seen_motion:.1f} px")
print(f"  Offsets agree to {mismatch:.1f} px")
ok = s1 >= 5 and s2 >= 3 and mismatch <= 2.5
print("\n=> CONFIRMED: same object, moving exactly as JPL predicts." if ok else
      "\n=> Only partly confirmed (see numbers above).")

# Zoomed picture for the demo / slides
cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
h = max(12, int(pred_motion) + 10)
sl = (slice(max(int(cy) - h, 0), int(cy) + h), slice(max(int(cx) - h, 0), int(cx) + h))
off = (sl[1].start, sl[0].start)

fig, ax = plt.subplots(1, 3, figsize=(16, 5.8))
for a, im, t in zip(ax, [TA, k * TB_al, TD],
                    [f"A  {Time(mjd[ia], format='mjd').iso[11:16]} UTC",
                     f"B  {Time(mjd[ib], format='mjd').iso[11:16]} UTC  (+{(mjd[ib]-mjd[ia])*24:.1f} h)",
                     "A - B   red = was here, blue = moved here"]):
    z = im[sl]
    if a is ax[2]:
        a.imshow(z, origin="lower", cmap="RdBu_r", vmin=-6 * sD, vmax=6 * sD)
    else:
        lo, hi = np.nanpercentile(z, [5, 99.5])
        a.imshow(z, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    a.plot(x1 - off[0], y1 - off[1], "o", mfc="none", mec="gold", ms=20, mew=2.5)
    a.plot(x2 - off[0], y2 - off[1], "o", mfc="none", mec="cyan", ms=20, mew=2.5)
    a.annotate("", xy=(x2 - off[0], y2 - off[1]), xytext=(x1 - off[0], y1 - off[1]),
               arrowprops=dict(arrowstyle="->", color="gold", lw=2))
    a.set_title(t, fontsize=12)
    a.axis("off")
fig.suptitle(f"SPHEREx sees asteroid {name} move "
             f"{seen_motion*6.15:.0f}\" in {(mjd[ib]-mjd[ia])*24:.1f} hours "
             f"(signal {s1:.0f} sigma / {s2:.0f} sigma)", fontsize=14)
plt.tight_layout()
plt.savefig("step5_asteroid_zoom.png", dpi=150)
plt.show()
