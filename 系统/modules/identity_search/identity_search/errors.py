"""Domain errors with concise messages and optional truthful diagnostics."""

from __future__ import annotations

from typing import Any, Mapping


class IdentitySearchError(RuntimeError):
    """Base class for an expected identity-search failure."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class InputValidationError(IdentitySearchError):
    """A command or library input violates the documented contract."""


class DependencyError(IdentitySearchError):
    """A required Python/runtime dependency is unavailable."""


class ModelPreflightError(IdentitySearchError):
    """The explicitly selected offline model did not pass preflight."""


class ProviderError(IdentitySearchError):
    """The requested ONNX Runtime execution provider is unavailable."""


class ImageDecodeError(IdentitySearchError):
    """An image file cannot be decoded."""


class NoFaceError(IdentitySearchError):
    """No face was detected in an image."""


class MultipleFacesError(IdentitySearchError):
    """More than one face was detected where exactly one is required."""


class InvalidEmbeddingError(IdentitySearchError):
    """The model did not return a usable face embedding."""


class InferenceError(IdentitySearchError):
    """The face model failed during inference."""


class EmptyLibraryError(IdentitySearchError):
    """No identity has a valid reference image."""
