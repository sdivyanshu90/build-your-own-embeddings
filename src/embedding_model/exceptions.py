"""Package-specific, safe-to-display errors."""


class EmbeddingModelError(Exception):
    """Base class for expected project errors."""


class ConfigurationError(EmbeddingModelError):
    """Configuration is invalid or internally incompatible."""


class DataValidationError(EmbeddingModelError):
    """Input data violates its declared schema."""


class ArtifactValidationError(EmbeddingModelError):
    """A model or index artifact is missing, incompatible, or tampered with."""


class ModelNotReadyError(EmbeddingModelError):
    """Inference was requested before the runtime was ready."""
