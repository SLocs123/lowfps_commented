#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2

import os
import subprocess

__all__ = ["configure_nccl", "configure_module"]


def configure_nccl():
    """Configure NCCL environment variables for multi-GPU training.

    The main pipeline in this repository does not call this directly, but it is
    part of the original YOLOX utility module and is kept for compatibility.
    """
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; popd > /dev/null"
    )
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"


def configure_module(ulimit_value=8192):
    """Apply a few process-wide settings that help computer-vision workloads.

    This mainly increases the open-file limit on Linux and disables OpenCL usage
    inside OpenCV so PyTorch data loading is less likely to compete with it.
    """
    try:
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, rlimit[1]))
    except Exception:
        # This can fail on platforms that do not expose `resource`, or when the
        # requested limit is higher than the system allows.
        pass

    # Prevent OpenCV from starting its own OpenCL thread pool.
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        # Some OpenCV builds do not expose these APIs.
        pass
