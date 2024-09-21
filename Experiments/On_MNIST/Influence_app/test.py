import torch

# Parameters for the Gaussian noise
mean = 0        # Mean of the distribution
std_dev = 1     # Standard deviation of the distribution
num_samples = 1000  # Number of samples

# Generate Gaussian noise
# torch.randn generates samples from a standard normal distribution (mean=0, std_dev=1)
noise = torch.randn(num_samples) * std_dev + mean

# Converting the tensor to a numpy array for plotting (if needed)
noise_np = noise.numpy()

# Example: Plotting the noise (if you want to visualize it)
import matplotlib.pyplot as plt

plt.hist(noise_np, bins=30, density=True)
plt.title("Histogram of Generated Gaussian Noise")
plt.xlabel("Noise Value")
plt.ylabel("Frequency")
plt.show()
