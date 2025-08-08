"""
Custom exceptions for OrthoReduce library.

This module defines custom exception classes for better error handling
and debugging throughout the library.
"""


class OrthoReduceError(Exception):
    """Base exception class for OrthoReduce library."""
    pass


class ValidationError(OrthoReduceError):
    """Raised when input validation fails."""
    pass


class ComputationError(OrthoReduceError):
    """Raised when computational operations fail."""
    pass


class ConfigurationError(OrthoReduceError):
    """Raised when configuration is invalid."""
    pass


class DimensionalityError(ValidationError):
    """Raised when dimensionality constraints are violated."""
    pass


class NumericalError(ComputationError):
    """Raised when numerical computation encounters issues."""
    pass