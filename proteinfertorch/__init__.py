import os

# Dynamically determine the root directory of the repository
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define other common paths default paths
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")

# Export these as part of the package API
__all__ = ["REPO_ROOT", "CONFIG_DIR", "CONFIG_FILE"]
