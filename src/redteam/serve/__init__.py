"""The serving layer: an HTTP interface over the trained defence and the red-team loop.

Three things live here and they answer different questions.

``bundle``
    A servable artefact. The detector, its guards, its threshold and the provenance of the
    run that produced them, saved atomically as one object so they cannot drift apart.

``store``
    SQLite persistence for runs and their alerts, so a client can page and filter without
    re-reading parquet on every request.

``api``
    A FastAPI application. ``POST /score`` is the one that matters - a payment in, a decision
    with reasons out - and the rest exist so a reviewer can drive the closed loop rather than
    read a finished report about it.

FastAPI is an optional dependency. Importing :mod:`redteam.serve.api` without it raises with
an instruction rather than a traceback, and nothing in the offline pipeline imports this
package, so a run that only generates and defends needs none of it installed.
"""

from .bundle import BUNDLE_VERSION, ServingBundle, from_artifacts  # noqa: F401
