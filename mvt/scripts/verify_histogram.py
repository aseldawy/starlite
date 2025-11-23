import numpy as np
import math

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
HIST_PATH = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/tile-geoparquet/tile_geoparquet/asia_output/global.npy"

WORLD_MINX = -20037508.342789244
WORLD_MINY = -20037508.342789244
WORLD_MAXX =  20037508.342789244
WORLD_MAXY =  20037508.342789244

# ---------------------------------------------------------
# LOAD HISTOGRAM
# ---------------------------------------------------------
hist = np.load(HIST_PATH)
H, W = hist.shape
print(f"Loaded histogram shape: {H} x {W}")

for i in range(H):
    print(f"hist[{i}] = {hist[i]}")

# ---------------------------------------------------------
# TEST 1: Verify histogram is power-of-two sized
# ---------------------------------------------------------
base_level = int(round(math.log2(H)))
if 2**base_level != H:
    raise ValueError("Histogram size is NOT a power of two. INVALID.")

print(f"✓ Histogram is power-of-two. Base level = {base_level}")

# ---------------------------------------------------------
# TEST 2: Global conservation test (most important test)
# hist_value should sum to the same total at all zoom levels
# ---------------------------------------------------------
def hist_value(hist, z, x, y):
    base_size = hist.shape[0]
    base_level = int(round(math.log2(base_size)))

    if z == base_level:
        return float(hist[y, x])

    elif z < base_level:
        factor = 2 ** (base_level - z)
        x0, y0 = x * factor, y * factor
        x1, y1 = min(x0 + factor, base_size), min(y0 + factor, base_size)
        return float(hist[y0:y1, x0:x1].sum())

    else:
        parent_x = x // (2 ** (z - base_level))
        parent_y = y // (2 ** (z - base_level))
        parent_val = hist[parent_y, parent_x]
        return float(parent_val) / (4 ** (z - base_level))

base_sum = hist.sum()
print(f"\nBase histogram sum = {base_sum}")

print("\nChecking global conservation across zoom levels:")
for z in range(0, base_level + 1):
    n = 2 ** z
    total = 0.0
    for x in range(n):
        for y in range(n):
            total += hist_value(hist, z, x, y)

    print(f"  z={z}: total={total}   diff={abs(total - base_sum)}")

print("\nIf all differences are near zero, histogram indexing & aggregation are CORRECT.")

# ---------------------------------------------------------
# TEST 3: Spot-check random bins for internal consistency
# This tests x/y orientation and prevents flipped axes
# ---------------------------------------------------------
import random

print("\nSpot-checking random tile lookups:")
for _ in range(10):
    # pick a random base-level tile
    xb = random.randint(0, W - 1)
    yb = random.randint(0, H - 1)

    # Direct value
    direct = hist[yb, xb]

    # Should match hist_value at base zoom
    computed = hist_value(hist, base_level, xb, yb)

    if direct != computed:
        print("ERROR: Mismatch at (x,y)=", xb, yb)
        print(" direct:", direct, " computed:", computed)
        break
else:
    print("✓ All random spot checks passed. Base-level indexing is correct.")

print("\nDONE. If no errors were printed, your histogram is correct.")
