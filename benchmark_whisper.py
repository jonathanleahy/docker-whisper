import whisper
import time
import numpy as np
import sys
import os
import torch


def benchmark_whisper(model_name, audio_path, num_runs=3, device='cpu'):
    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name, device=device)

    print(f"Audio file: {audio_path}")
    print(f"Number of runs: {num_runs}")
    print(f"Device: {device}")

    times = []
    for i in range(num_runs):
        print(f"\nRun {i + 1}/{num_runs}")
        start_time = time.time()
        result = model.transcribe(audio_path)
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        print(f"Run {i + 1} time: {run_time:.2f} seconds")
        print(f"Transcription: {result['text'][:100]}...")  # Print first 100 characters

    avg_time = np.mean(times)
    std_dev = np.std(times)

    print(f"\nResults for {model_name} model:")
    print(f"Average processing time: {avg_time:.2f} seconds")
    print(f"Standard deviation: {std_dev:.2f} seconds")
    print(f"Min time: {min(times):.2f} seconds")
    print(f"Max time: {max(times):.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python benchmark_whisper.py <model_name> <audio_path> [num_runs] [device]")
        sys.exit(1)

    model_name = sys.argv[1]
    audio_path = sys.argv[2]
    num_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    device = sys.argv[4] if len(sys.argv) > 4 else 'cpu'

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'

    benchmark_whisper(model_name, audio_path, num_runs, device)
