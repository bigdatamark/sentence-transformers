# [PACKAGE DOC]
# Training callbacks for sparse encoders. Re-exports SpladeRegularizerWeightSchedulerCallback and
# the SchedulerType enum used to warm up SpladeLoss regularizer weights.
from __future__ import annotations

from .splade_callbacks import SchedulerType, SpladeRegularizerWeightSchedulerCallback

__all__ = ["SpladeRegularizerWeightSchedulerCallback", "SchedulerType"]
