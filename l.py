import numpy as np

# Load your existing (30, 1, 64, 63) array and move the channel axis to the end -> (30, 64, 63, 1)
cal = np.load("calibration_data.npy")
cal_nhwc = np.transpose(cal, (0, 2, 3, 1)).astype(np.float32)
np.save("calibration_data.npy", cal_nhwc)

print("Updated calibration_data.npy shape:", cal_nhwc.shape)
