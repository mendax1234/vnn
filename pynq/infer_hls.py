from pynq import Overlay, allocate
from pathlib import Path
import numpy as np
import time
import sys

# ── Load overlay ─────────────────────────────────────────────────────────────
overlay = Overlay("/home/ubuntu/hls/cnn_accelerator.xsa")
dma     = overlay.axi_dma_0
ip      = overlay.cnn_accelerator_0   # packaged from hls/cnn_accelerator.cpp


# ── CIFAR-10 constants ───────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
CIFAR_BIN  = REPO_ROOT / "data/test_batch.bin"
PARAM_BIN  = REPO_ROOT / "model/weights_q6_10.bin"

DEFAULT_NUM_IMAGES = 10000
if len(sys.argv) > 1:
    NUM_IMAGES = int(sys.argv[1])
else:
    NUM_IMAGES = DEFAULT_NUM_IMAGES

# NUM_IMAGES = 10000
IMG_PIXELS = 3 * 32 * 32   # 3072 pixels per image

if not 1 <= NUM_IMAGES <= 10000:
    raise ValueError("NUM_IMAGES must be between 1 and 10000")

N_W_CONV1 = 32 * 3  * 3 * 3
N_W_CONV2 = 32 * 32 * 3 * 3
N_W_CONV3 = 64 * 32 * 3 * 3
N_W_CONV4 = 64 * 64 * 3 * 3
N_W_CONV5 = 128 * 64 * 3 * 3
N_W_CONV6 = 128 * 128 * 3 * 3
N_W_FC    = 10 * 128

N_B_CONV1 = 32
N_B_CONV2 = 32
N_B_CONV3 = 64
N_B_CONV4 = 64
N_B_CONV5 = 128
N_B_CONV6 = 128
N_B_FC    = 10

NUM_PARAM_WORDS = (
    N_W_CONV1 + N_W_CONV2 + N_W_CONV3 + N_W_CONV4 + N_W_CONV5 + N_W_CONV6 + N_W_FC +
    N_B_CONV1 + N_B_CONV2 + N_B_CONV3 + N_B_CONV4 + N_B_CONV5 + N_B_CONV6 + N_B_FC
)

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]

def float_to_q6_10_int16(arr_f32: np.ndarray) -> np.ndarray:
    """
    Convert float array to signed int16 carrying q6.10 fixed-point bit pattern.
    Clipping range for ap_fixed<16,6> is [-32.0, 31.9990234375].
    """
    clipped = np.clip(arr_f32, -32.0, 31.9990234375)
    scaled  = np.round(clipped * 1024.0).astype(np.int16)
    return scaled

# ── Load precomputed parameter blob ──────────────────────────────────────────
params = np.fromfile(PARAM_BIN, dtype="<i2")

if params.size != NUM_PARAM_WORDS:
    raise RuntimeError(
        f"Parameter blob size mismatch: got {params.size} words, "
        f"expected {NUM_PARAM_WORDS}"
    )

print(f"Loaded parameter blob: {params.size} int16 words")



# ── Load and normalise CIFAR-10 test batch ───────────────────────────────────
# Each record: 1 byte label + 3072 bytes pixels (channels-first)
raw = np.fromfile(CIFAR_BIN, dtype=np.uint8, count=NUM_IMAGES * 3073)
raw = raw.reshape(NUM_IMAGES, 3073)

labels = raw[:, 0].astype(np.int32)
pixels = raw[:, 1:].reshape(NUM_IMAGES, 3, 32, 32).astype(np.float32) / 255.0

# Normalise per channel
for c in range(3):
    pixels[:, c, :, :] = (pixels[:, c, :, :] - MEAN[c]) / STD[c]

# Convert to ap_fixed<16,6> bit pattern
pixels_q = float_to_q6_10_int16(pixels)

# Flatten to (NUM_IMAGES * 3072,) — channels-first matches HLS loop order
flat_input = pixels_q.reshape(-1)  # shape: (30,720,000,)

expected_img_words = NUM_IMAGES * IMG_PIXELS
if flat_input.size != expected_img_words:
    raise RuntimeError(
        f"Flattened image size mismatch: got {flat_input.size}, "
        f"expected {expected_img_words}"
    )

print(f"Loaded image payload:  {flat_input.size} int16 words")

# ── Concatenate parameters first, then images ────────────────────────────────
full_input = np.concatenate([params, flat_input]).astype(np.int16, copy=False)
expected_total_words = NUM_PARAM_WORDS + expected_img_words

if full_input.size != expected_total_words:
    raise RuntimeError(
        f"Total input size mismatch: got {full_input.size}, "
        f"expected {expected_total_words}"
    )

print(f"Total DMA input size:  {full_input.size} int16 words")
print(f"Output size:           {NUM_IMAGES} int16 words")

# ── Allocate DMA buffers ─────────────────────────────────────────────────────
in_buf = out_buf = None

try:
    in_buf  = allocate(shape=(full_input.size,), dtype=np.int16)
    out_buf = allocate(shape=(NUM_IMAGES,),      dtype=np.int16)

    in_buf[:]  = full_input
    out_buf[:] = 0

    in_buf.flush()
    out_buf.flush()

    # ── Set num_images via AXI Lite before starting ──────────────────────────────
    ip.write(ip.register_map.num_images.address, NUM_IMAGES)

    dma.recvchannel.stop()
    dma.sendchannel.stop()
    dma.recvchannel.start()
    dma.sendchannel.start()


    # ── Run inference ────────────────────────────────────────────────────────────
    print(f"Running inference on {NUM_IMAGES} images...")
    start = time.perf_counter_ns()

    dma.recvchannel.transfer(out_buf)
    ip.register_map.CTRL.AP_START = 1
    dma.sendchannel.transfer(in_buf)

    dma.sendchannel.wait()
    dma.recvchannel.wait()

    end = time.perf_counter_ns()
    elapsed_ms = (end - start) / 1e6

    # Invalidate output cache after DMA
    out_buf.invalidate()
    dma.sendchannel.stop()
    dma.recvchannel.stop()

    predictions = out_buf[:].astype(np.int32)
    
finally: 
    if in_buf is not None:
        in_buf.freebuffer()
    if out_buf is not None:
        out_buf.freebuffer()

# ── Accuracy ─────────────────────────────────────────────────────────────────

correct     = np.sum(predictions == labels)
accuracy    = 100.0 * correct / NUM_IMAGES

print(f"\nTop-1 accuracy: {correct} / {NUM_IMAGES} = {accuracy:.2f}%")
print(f"Total time:     {elapsed_ms:.1f} ms")
print(f"Per image:      {elapsed_ms / NUM_IMAGES * 1000:.1f} µs")
print(f"Throughput:     {NUM_IMAGES / (elapsed_ms / 1000):.0f} images/sec")

# ── Per-class breakdown ──────────────────────────────────────────────────────
print(f"\n{'Class':<12}  {'Correct':>7}  {'Total':>7}  {'Acc':>7}")
print("─" * 40)
for c in range(10):
    mask = (labels == c)
    n    = int(mask.sum())
    cor  = int((predictions[mask] == c).sum())
    acc  = (100.0 * cor / n) if n > 0 else 0.0
    print(f"{CLASS_NAMES[c]:<12}  {cor:>7}  {n:>7}  {acc:>6.2f}%")

# ── Sanity checks ────────────────────────────────────────────────────────────
invalid = np.sum((predictions < 0) | (predictions > 9))
if invalid > 0:
    print(f"\nWARNING: {invalid} predictions out of range 0-9")

print(f"First 10 predictions: {predictions[:10].tolist()}")
print(f"First 10 labels:      {labels[:10].tolist()}")

