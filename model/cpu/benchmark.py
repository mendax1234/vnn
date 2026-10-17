"""V2 CPU inference benchmark; see SOFTWARE_V2.md for KV260 build instructions."""
import argparse
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
RECORD_BYTES = 3073


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("num_images", nargs="?", type=positive_int, default=100)
    parser.add_argument("--mode", choices=("fixed", "fp32"), default="fixed",
                        help="CPU arithmetic (default: fixed)")
    parser.add_argument("--binary", type=Path, help="override the selected mode's executable")
    parser.add_argument("--params", type=Path,
                        default=Path("/home/ubuntu/weights_biases_v2_int16.bin"),
                        help="exact FPGA parameter blob, used by fixed mode")
    parser.add_argument("--cifar", type=Path, default=Path("/home/ubuntu/test_batch.bin"))
    parser.add_argument("--warmup", type=positive_int, default=1,
                        help="untimed inferences before measurement (default: 1)")
    parser.add_argument("--repeats", type=positive_int, default=1,
                        help="timed passes in the same C++ process (default: 1)")
    args = parser.parse_args()
    if args.num_images > 10000:
        parser.error("num_images must be between 1 and 10000")
    binary = (args.binary or Path("/home/ubuntu/software_implementation/" +
              ("software_model_fixed" if args.mode == "fixed" else "software_model"))).expanduser().resolve()
    cifar = args.cifar.expanduser().resolve()
    if not binary.is_file():
        parser.error("CPU binary not found: {}. Build the v2 sources on the board.".format(binary))
    params = args.params.expanduser().resolve()
    if args.mode == "fixed" and (not params.is_file() or params.stat().st_size != 288298 * 2):
        parser.error("Fixed mode needs a 288298-word int16 parameter file: {}".format(params))
    if not cifar.is_file():
        parser.error("CIFAR file not found: {}".format(cifar))
    if cifar.stat().st_size % RECORD_BYTES:
        parser.error("CIFAR file contains an incomplete 3073-byte record")
    with cifar.open("rb") as stream:
        raw = stream.read(args.num_images * RECORD_BYTES)
    if len(raw) != args.num_images * RECORD_BYTES:
        parser.error("CIFAR file has fewer than {} images".format(args.num_images))
    labels = list(raw[::RECORD_BYTES])
    if any(label > 9 for label in labels):
        parser.error("CIFAR labels must be in the range 0-9")

    print("Running V2 {} CPU inference (single-threaded): {} images, {} warm-up(s), {} pass(es)".format(
        args.mode, args.num_images, args.warmup, args.repeats), flush=True)
    command = [str(binary), str(args.num_images), str(cifar), str(args.warmup), str(args.repeats)]
    if args.mode == "fixed":
        command.append(str(params))
    start = time.perf_counter()
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    process_ms = (time.perf_counter() - start) * 1000.0
    if result.returncode:
        raise RuntimeError("{} exited with code {}:\n{}\n{}".format(
            binary.name, result.returncode, result.stderr, result.stdout))
    predictions = None
    times = []
    arithmetic = None
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "ARITHMETIC":
            if len(fields) != 2 or arithmetic is not None:
                raise RuntimeError("Malformed or duplicate arithmetic identifier")
            arithmetic = fields[1]
        elif fields[0] == "PREDICTIONS":
            if predictions is not None:
                raise RuntimeError("Duplicate PREDICTIONS output")
            predictions = [int(value) for value in fields[1:]]
        elif fields[0] == "INFERENCE_TIME_MS":
            if len(fields) != 2:
                raise RuntimeError("Malformed inference timing output")
            times.append(float(fields[1]))
    if args.mode == "fixed" and arithmetic != "FIXED_Q6_10":
        raise RuntimeError("Fixed mode requires the new fixed-point executable; arithmetic identifier missing")
    if args.mode == "fp32" and arithmetic is not None:
        raise RuntimeError("FP32 mode received a different arithmetic executable")
    if predictions is None or len(predictions) != args.num_images:
        raise RuntimeError("Missing predictions or incorrect count; rebuild the v2 binary")
    if any(prediction < 0 or prediction > 9 for prediction in predictions):
        raise RuntimeError("Predictions must be in the range 0-9")
    if len(times) != args.repeats:
        raise RuntimeError("Missing inference timings; rebuild the updated v2 C++ sources")
    if any(not math.isfinite(value) or value <= 0 for value in times):
        raise RuntimeError("Inference times must be finite and positive")

    elapsed_ms = statistics.median(times)
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    print("\nTop-1 accuracy: {} / {} = {:.2f}%".format(
        correct, args.num_images, 100.0 * correct / args.num_images))
    print("Timing scope:   inference only; median of {} pass(es)".format(args.repeats))
    print("Total time:     {:.3f} ms".format(elapsed_ms))
    print("Per image:      {:.3f} µs".format(elapsed_ms * 1000.0 / args.num_images))
    print("Throughput:     {:.3f} images/sec".format(args.num_images * 1000.0 / elapsed_ms))
    print("Inference FPS:  {:.3f}".format(args.num_images * 1000.0 / elapsed_ms))
    for index, value in enumerate(times, 1):
        print("Pass {}: {:.3f} ms, {:.3f} FPS".format(
            index, value, args.num_images * 1000.0 / value))
    print("Process time:   {:.3f} ms (loading, warm-up, all passes, reporting)".format(process_ms))
    print("\n{:<12} {:>7} {:>7} {:>8}".format("Class", "Correct", "Total", "Acc (%)"))
    for class_id, name in enumerate(CLASS_NAMES):
        total = labels.count(class_id)
        hits = sum(prediction == label == class_id
                   for prediction, label in zip(predictions, labels))
        print("{:<12} {:>7} {:>7} {:>8.2f}".format(
            name, hits, total, 100.0 * hits / total if total else 0.0))
    print("First 10 predictions: {}".format(predictions[:10]))
    print("First 10 labels:      {}".format(labels[:10]))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
