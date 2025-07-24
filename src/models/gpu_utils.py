"""GPU Utilities for Climate ML Pipeline.

This module provides utilities for detecting and utilizing GPU acceleration
in the climate temperature prediction models.
"""

import logging
import os
import platform
from typing import Any, Dict

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU detection and configuration for ML models."""

    def __init__(self):
        """Initialize GPU manager."""
        self._gpu_available = None
        self._cuda_available = None
        self._torch_available = None
        self._gpu_info = None

    def is_gpu_available(self) -> bool:
        """Check if GPU is available for computation.

        Returns:
            True if GPU is available and accessible
        """
        if self._gpu_available is not None:
            return self._gpu_available

        gpu_checks = [
            self._check_torch_cuda(),
            self._check_nvidia_smi(),
            self._check_cuda_env(),
        ]

        self._gpu_available = any(gpu_checks)
        logger.info(f"GPU availability check: {self._gpu_available}")

        return self._gpu_available

    def _check_torch_cuda(self) -> bool:
        """Check if PyTorch with CUDA is available."""
        try:
            import torch

            self._torch_available = True
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = (
                    torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                )
                logger.info(
                    f"PyTorch CUDA available: {device_count} GPU(s), Primary: {device_name}"
                )
                self._cuda_available = True
                return True
        except ImportError:
            self._torch_available = False
            logger.debug("PyTorch not available")
        except Exception as e:
            logger.debug(f"PyTorch CUDA check failed: {e}")

        return False

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi command is available."""
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_lines = [
                    line for line in result.stdout.split("\n") if "GPU" in line
                ]
                logger.info(f"nvidia-smi detected {len(gpu_lines)} GPU(s)")
                return len(gpu_lines) > 0
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            logger.debug("nvidia-smi not available or failed")
        except Exception as e:
            logger.debug(f"nvidia-smi check failed: {e}")

        return False

    def _check_cuda_env(self) -> bool:
        """Check for CUDA environment variables."""
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            logger.info(f"CUDA environment detected: {cuda_home}")
            return True
        return False

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get detailed GPU information.

        Returns:
            Dictionary with GPU information
        """
        if self._gpu_info is not None:
            return self._gpu_info

        info = {
            "gpu_available": self.is_gpu_available(),
            "torch_available": self._torch_available,
            "cuda_available": self._cuda_available,
            "platform": platform.platform(),
            "devices": [],
        }

        if self._torch_available and self._cuda_available:
            try:
                import torch

                for i in range(torch.cuda.device_count()):
                    device_info = {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": torch.cuda.get_device_properties(
                            i
                        ).total_memory,
                        "memory_free": torch.cuda.memory_reserved(i),
                        "compute_capability": torch.cuda.get_device_properties(i).major,
                    }
                    info["devices"].append(device_info)
            except Exception as e:
                logger.warning(f"Failed to get detailed GPU info: {e}")

        self._gpu_info = info
        return info

    def get_xgboost_params(self) -> Dict[str, Any]:
        """Get XGBoost parameters optimized for available hardware.

        Returns:
            Dictionary of XGBoost parameters
        """
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

        if self.is_gpu_available():
            try:
                # Test if GPU tree method is available
                import xgboost as xgb

                # Check XGBoost version and GPU support
                if hasattr(xgb, "XGBRegressor"):
                    params.update(
                        {
                            "tree_method": "gpu_hist",
                            "gpu_id": 0,
                            "predictor": "gpu_predictor",
                        }
                    )
                    logger.info("XGBoost configured for GPU acceleration")
                else:
                    logger.warning("XGBoost GPU support not available, using CPU")
            except Exception as e:
                logger.warning(f"Failed to configure XGBoost for GPU: {e}")

        return params

    def get_lightgbm_params(self) -> Dict[str, Any]:
        """Get LightGBM parameters optimized for available hardware.

        Returns:
            Dictionary of LightGBM parameters
        """
        params = {
            "objective": "regression",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        if self.is_gpu_available():
            try:
                # Try to enable GPU support (lightgbm import handled elsewhere)
                params.update(
                    {"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
                )
                logger.info("LightGBM configured for GPU acceleration")
            except Exception as e:
                logger.warning(f"Failed to configure LightGBM for GPU: {e}")
                # Fallback to CPU
                params["device_type"] = "cpu"

        return params

    def get_recommended_setup(self) -> str:
        """Get recommendations for GPU setup.

        Returns:
            String with setup recommendations
        """
        if self.is_gpu_available():
            return "GPU detected and ready for acceleration!"

        recommendations = [
            "No GPU detected. For GPU acceleration:",
            "1. Install PyTorch with CUDA: 'pip install torch --index-url https://download.pytorch.org/whl/cu121'",
            "2. For RunPod containers, use a CUDA-enabled image",
            "3. Ensure NVIDIA drivers are installed",
            "4. Install GPU requirements: 'pip install -r requirements/gpu.txt'",
        ]

        return "\n".join(recommendations)


# Global GPU manager instance
gpu_manager = GPUManager()
