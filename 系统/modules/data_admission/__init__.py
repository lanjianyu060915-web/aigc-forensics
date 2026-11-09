"""Explicit-manifest data admission; never discover dataset files by scanning."""
from .admission import AdmissionError, AdmittedDataset, DataSnapshot, PolicyAuthority, create_receipt

__all__ = ["AdmissionError", "AdmittedDataset", "DataSnapshot", "PolicyAuthority", "create_receipt"]
