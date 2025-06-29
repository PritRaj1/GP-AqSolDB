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
    'cupy-cuda12x', # Replace with your CUDA version
]

for package in requirements:
    try:
        print(f"Installing {package}...")
        install(package)
        print(f"{package} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}.")

if __name__ == "__main__":
    for r in requirements:
        print(r)
