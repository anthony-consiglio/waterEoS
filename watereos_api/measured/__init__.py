"""Measured water data for the web app.

The curated experimental database (``fitting_data/``, kept outside this
repository) is copied into a SQL database by ``python -m
watereos_api.measured.load``; the API in ``watereos_api.routes.measured``
serves it read-only. See ``docs/measured_data.md``.
"""
