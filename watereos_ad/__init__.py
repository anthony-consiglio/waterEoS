"""
Automatic differentiation backend for waterEoS using JAX.

Provides JAX-based implementations of the two-state EoS models,
replacing hand-coded derivatives with jax.grad and hand-vectorized
batch code with jax.vmap.

JAX is an optional dependency. When not installed, models fall back
to their hand-coded core.py implementations.
"""

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None
