import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load your ECG CSV file
df = pd.read_csv("3.csv")   # make sure 2.csv is in the same folder
signal = df["data 0"].values

# Create a sample index as time (assuming 250 Hz if unknown)
fs = 250  # Hz (adjust if you know your sampling rate)
t = np.arange(len(signal)) / fs

# Bandpass filter (0.5–40 Hz typical ECG range)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

filtered = bandpass_filter(signal, 0.5, 40.0, fs)

# Plot raw vs filtered
plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(t[:2000], signal[:2000], color='gray')
plt.title("Raw ECG (first 2000 samples)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2,1,2)
plt.plot(t[:2000], filtered[:2000], color='blue')
plt.title("Filtered ECG (0.5–40 Hz Bandpass)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
