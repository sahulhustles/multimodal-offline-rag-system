"""Pytest configuration and environment setup for backend tests."""
import os

# Prevent transformers from loading tensorflow and encountering protobuf version mismatch
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
