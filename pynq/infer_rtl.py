from pathlib import Path
import time

import numpy as np
from pynq import Overlay, allocate


# Copy the overlay, parameters, and dataset to the KV260 at these paths.
OVERLAY_PATH = "/home/ubuntu/hdl/cnn_dataflow.xsa"
PARAM_BIN = "/home/ubuntu/weights_q6_10.bin"
CIFAR_BIN = "/home/ubuntu/test_batch.bin"

# Start with one image. Increase this only after a single-image test passes.
NUM_IMAGES = 1

IMG_PIXELS = 3 * 32 * 32
NUM_PARAM_WORDS = 288_298

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def float_to_q6_10_int16(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -32.0, 31.9990234375)
    return np.round(values * 1024.0).astype(np.int16)


def find_ip(overlay: Overlay, preferred_name: str, type_fragment: str):
    if hasattr(overlay, preferred_name):
        return getattr(overlay, preferred_name)

    matches = [
        name for name, description in overlay.ip_dict.items()
        if type_fragment.lower() in description.get("type", "").lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely find {type_fragment!r}. "
            f"Available IP: {list(overlay.ip_dict)}"
        )
    return getattr(overlay, matches[0])


def main() -> None:
    for path in (OVERLAY_PATH, PARAM_BIN, CIFAR_BIN):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    overlay = Overlay(OVERLAY_PATH)
    dma = find_ip(overlay, "axi_dma_0", "axi_dma")
    cnn = find_ip(overlay, "cnn_v2_accelerator_0", "cnn_v2_accelerator")

    params = np.fromfile(PARAM_BIN, dtype="<i2")
    if params.size != NUM_PARAM_WORDS:
        raise RuntimeError(
            f"V2 parameter count mismatch: got {params.size}, "
            f"expected {NUM_PARAM_WORDS}"
        )

    raw = np.fromfile(CIFAR_BIN, dtype=np.uint8, count=NUM_IMAGES * 3073)
    if raw.size != NUM_IMAGES * 3073:
        raise RuntimeError(
            f"CIFAR file contains only {raw.size // 3073} complete requested records"
        )
    raw = raw.reshape(NUM_IMAGES, 3073)

    labels = raw[:, 0].astype(np.int32)
    pixels = raw[:, 1:].reshape(NUM_IMAGES, 3, 32, 32).astype(np.float32)
    pixels /= 255.0
    pixels = (pixels - MEAN[None, :, None, None]) / STD[None, :, None, None]
    image_words = float_to_q6_10_int16(pixels).reshape(-1)

    full_input = np.concatenate((params, image_words)).astype(np.int16, copy=False)

    in_buf = None
    out_buf = None
    try:
        in_buf = allocate(shape=full_input.shape, dtype=np.int16)
        out_buf = allocate(shape=(NUM_IMAGES,), dtype=np.int16)
        in_buf[:] = full_input
        out_buf[:] = 0
        in_buf.flush()
        out_buf.flush()

        cnn.register_map.num_images = NUM_IMAGES

        print(f"Running CNN V2 on {NUM_IMAGES} image(s)")
        print(f"Input: {full_input.size} words; output: {NUM_IMAGES} words")
        start_ns = time.perf_counter_ns()

        # Arm the output path before the accelerator can produce a result.
        dma.recvchannel.transfer(out_buf)
        cnn.register_map.CTRL.AP_START = 1
        dma.sendchannel.transfer(in_buf)

        dma.sendchannel.wait()
        dma.recvchannel.wait()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6

        out_buf.invalidate()
        predictions = np.asarray(out_buf, dtype=np.int16).astype(np.int32)
    finally:
        if in_buf is not None:
            in_buf.freebuffer()
        if out_buf is not None:
            out_buf.freebuffer()

    correct = int(np.sum(predictions == labels))
    print(f"Accuracy: {correct}/{NUM_IMAGES} = {100.0 * correct / NUM_IMAGES:.2f}%")
    print(f"Elapsed: {elapsed_ms:.1f} ms ({elapsed_ms / NUM_IMAGES:.1f} ms/image)")
    for index, (prediction, label) in enumerate(zip(predictions, labels)):
        predicted_name = (
            CLASS_NAMES[prediction] if 0 <= prediction < len(CLASS_NAMES) else "INVALID"
        )
        print(
            f"{index}: prediction={prediction} ({predicted_name}), "
            f"label={label} ({CLASS_NAMES[label]})"
        )


if __name__ == "__main__":
    main()
