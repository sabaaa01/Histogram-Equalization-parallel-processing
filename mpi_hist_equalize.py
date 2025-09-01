#!/usr/bin/env python3
"""
MPI-parallel Histogram Equalization for Color Images (L = 256 buckets)
Equalizes the luminance channel in LAB color space to preserve color information.

Usage:
    mpirun -np 4 python3 mpi_color_hist_eq.py input.jpg output.jpg

Requirements:
    pip install mpi4py numpy opencv-python matplotlib
"""
from mpi4py import MPI
import numpy as np
import cv2
import sys
import time
import os  # added for histogram folder creation

# Try to import matplotlib, make it optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Histogram plots will be skipped.")

L = 256  # number of intensity levels

def compute_histogram(chunk):
    if chunk.size == 0:
        return np.zeros(L, dtype=np.int64)
    flat = chunk.ravel()
    hist = np.bincount(flat, minlength=L).astype(np.int64)
    return hist

def plot_histograms(original_hist, equalized_hist, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(L), original_hist, width=1, alpha=0.7, color='blue', edgecolor='none')
    ax1.set_title('Original Image Histogram (L Channel)')
    ax1.set_xlabel('Intensity Level')
    ax1.set_ylabel('Pixel Count')
    ax1.set_xlim(0, L-1)
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(L), equalized_hist, width=1, alpha=0.7, color='red', edgecolor='none')
    ax2.set_title('Equalized Image Histogram (L Channel)')
    ax2.set_xlabel('Intensity Level')
    ax2.set_ylabel('Pixel Count')
    ax2.set_xlim(0, L-1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram comparison: {output_path}")

def compute_lut_from_hist(hist, total_pixels):
    pdf = hist.astype(np.float64) / float(total_pixels)
    cdf = np.cumsum(pdf)
    lut = np.floor((L - 1) * cdf + 0.5).astype(np.uint8)
    return lut

def rgb_to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

def lab_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

def sequential_equalize_color(img):
    lab_img = rgb_to_lab(img)
    l_channel = lab_img[:, :, 0]
    M, N = l_channel.shape
    MN = M * N
    hist = compute_histogram(l_channel)
    lut = compute_lut_from_hist(hist, MN)
    equalized_l = lut[l_channel]
    result_lab = lab_img.copy()
    result_lab[:, :, 0] = equalized_l
    result = lab_to_rgb(result_lab)
    return result, hist, lut

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if len(sys.argv) < 3:
            print("Usage: mpirun -np <procs> python3 mpi_color_hist_eq.py input.jpg output.jpg")
            sys.exit(1)
        in_path = sys.argv[1]
        out_path = sys.argv[2]

        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: failed to read input image '{in_path}'")
            sys.exit(1)

        lab_img = rgb_to_lab(img)
        l_channel = lab_img[:, :, 0]
        H, W = l_channel.shape
        chunks = np.array_split(l_channel, size, axis=0)
        lab_chunks = np.array_split(lab_img, size, axis=0)
        total_pixels = H * W
    else:
        img = lab_img = out_path = chunks = lab_chunks = None
        H = W = total_pixels = None

    H = comm.bcast(H, root=0)
    W = comm.bcast(W, root=0)
    total_pixels = comm.bcast(total_pixels, root=0)
    out_path = comm.bcast(out_path, root=0)

    my_l_chunk = comm.scatter(chunks, root=0)
    my_lab_chunk = comm.scatter(lab_chunks, root=0)

    comm.Barrier()
    t_start = MPI.Wtime()

    local_hist = compute_histogram(my_l_chunk)
    global_hist = np.zeros(L, dtype=np.int64) if rank == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)

    if rank == 0:
        if total_pixels == 0:
            print("Error: empty image")
            sys.exit(1)
        lut = compute_lut_from_hist(global_hist, total_pixels)
    else:
        lut = None

    lut = comm.bcast(lut, root=0)

    if my_l_chunk.size == 0:
        my_equalized_lab = my_lab_chunk
    else:
        equalized_l = lut[my_l_chunk]
        my_equalized_lab = my_lab_chunk.copy()
        my_equalized_lab[:, :, 0] = equalized_l

    gathered_lab = comm.gather(my_equalized_lab, root=0)

    comm.Barrier()
    t_end = MPI.Wtime()
    parallel_time = t_end - t_start

    if rank == 0:
        result_lab = np.vstack(gathered_lab)
        result_img = lab_to_rgb(result_lab)
        cv2.imwrite(out_path, result_img)
        print(f"Saved parallel equalized color image: {out_path}")
        print(f"Processes: {size}, Image: {H}x{W}, Total pixels: {total_pixels}")
        print(f"Parallel (end-to-end) time: {parallel_time:.6f} seconds")

        seq_start = time.time()
        seq_result, seq_hist, seq_lut = sequential_equalize_color(img)
        seq_end = time.time()
        seq_time = seq_end - seq_start
        seq_out = out_path.replace(".", "_seq.", 1) if "." in out_path else out_path + "_seq.jpg"
        cv2.imwrite(seq_out, seq_result)
        print(f"Sequential time (same algo): {seq_time:.6f} seconds")
        if seq_time > 0:
            print(f"Speedup (sequential / parallel): {seq_time / parallel_time:.3f}x")

        # Verification
        result_lab_check = rgb_to_lab(result_img)
        seq_lab_check = rgb_to_lab(seq_result)
        if np.array_equal(result_lab_check[:, :, 0], seq_lab_check[:, :, 0]):
            print("Verification: parallel L channel == sequential L channel ✅")
        else:
            same = np.mean(result_lab_check[:, :, 0] == seq_lab_check[:, :, 0])
            print(f"Verification: fraction of identical L pixels = {same:.6f}")

        # -------------------------------
        # Generate and save histograms
        # -------------------------------
        if MATPLOTLIB_AVAILABLE:
            hist_dir = "histograms"
            os.makedirs(hist_dir, exist_ok=True)

            input_l = rgb_to_lab(img)[:, :, 0]
            output_l = rgb_to_lab(result_img)[:, :, 0]

            input_hist = compute_histogram(input_l)
            output_hist = compute_histogram(output_l)

            hist_path = os.path.join(hist_dir, "histogram.jpg")
            plot_histograms(input_hist, output_hist, hist_path)

    comm.Barrier()

if __name__ == "__main__":
    main()

#mpirun -np 4 python mpi_hist_equalize.py test.jpg output.jpg
