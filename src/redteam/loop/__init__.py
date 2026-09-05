"""Closed-loop co-evolution between two economically motivated adversaries.

The pieces, in the order a round uses them:

* :mod:`~redteam.loop.surrogate` - red steals a copy of the detector from approve/decline
  answers, then optimises against the copy.
* :mod:`~redteam.loop.strategy` - the structural genes, which change the plan rather than
  the appearance of a campaign.
* :mod:`~redteam.loop.economics` - what each lever costs and what each side is optimising,
  which is profit and total cost of ownership rather than recall.
* :mod:`~redteam.loop.incremental` - recompute only the rows a mutation touched, which is
  what makes a twelve-round search affordable.
* :mod:`~redteam.loop.blue` - the defender's priced move set: hold, re-threshold, add a
  rule, add friction, retrain.
* :mod:`~redteam.loop.discovery` - the return path, feeding surviving tactics back into the
  taxonomy and reporting which library controls they walked through.
"""

from .blue import BlueMove, Rule, choose_move  # noqa: F401
from .coevolution import (  # noqa: F401
    AttackGenome,
    LoopResult,
    RoundResult,
    aged_account_pool,
    apply_genome,
    convergence,
    run_loop,
)
from .discovery import control_gap, control_gap_summary, transfer_matrix  # noqa: F401
from .economics import AttackerCosts, DefenderCosts, score_campaign  # noqa: F401
from .strategy import StrategyGenes, apply_strategy, random_genes  # noqa: F401
from .surrogate import SurrogateOracle  # noqa: F401
