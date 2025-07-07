import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

requirements = [
    'numpy',
    'scipy',
    'scikit-learn',
    'optuna',
    'pytest',
    'matplotlib',
    'seaborn',
    'pandas',
    'imageio',

    # Replace with your CUDA version
    'cupy-cuda12x', 
]

# Install JAX separately
jax_gpu_packages = [
    'jax[cuda12_pip]',
    'jaxlib[cuda12_pip]',
]

for package in requirements:
    try:
        print(f"Installing {package}...")
        install(package)
        print(f"{package} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}.")

print("Installing JAX with GPU support...")
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--upgrade", "jax[cuda12_pip]", 
        "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    ])
    print("JAX with GPU support installed successfully.")
except subprocess.CalledProcessError:
    print("Failed to install JAX with GPU support. Falling back to CPU version.")
    try:
        install('jax')
        print("JAX CPU version installed as fallback.")
    except subprocess.CalledProcessError:
        print("Failed to install JAX.")

if __name__ == "__main__":
    for r in requirements:
        print(r)
