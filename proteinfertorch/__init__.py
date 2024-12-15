import os

# Dynamically determine the root directory of the repository
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)

# Define other common paths default paths
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")

# Export these as part of the package API
__all__ = ["REPO_ROOT","PACKAGE_ROOT", "CONFIG_DIR", "CONFIG_FILE"]
