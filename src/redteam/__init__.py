"""Red-team / blue-team payment fraud simulation and detection pipeline.

Three pillars, one feedback loop:

* ``redteam.identify`` - a validated taxonomy of GenAI-enabled payment fraud vectors.
* ``redteam.generate`` - entity, benign-traffic and attack simulators that turn the
  taxonomy into labelled, high-fidelity payment data.
* ``redteam.defend``   - feature engineering, detection models and the guards that
  cover attack classes a tabular classifier alone cannot see.

``redteam.loop`` closes the circuit: the attacker mutates against the deployed model,
the defender retrains on what got through, and both are measured round over round.
"""

__version__ = "1.0.0"

from .config import Config  # noqa: F401
