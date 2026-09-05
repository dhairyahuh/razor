"""Command line entry point: ``python -m redteam <stage>``.

Stages can be run individually or as a single ``run`` that threads one config through all
three pillars and the closed loop. Intermediate data is written to ``data/`` so ``defend``
and ``loop`` can be re-run against a fixed dataset without regenerating it, which matters
when iterating on the model.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import ROOT, Config
from .defend.pipeline import run_defence
from .genai.cache import CacheMiss, LLMCache
from .genai.ideate import propose_vectors, to_yaml_block
from .genai.payloads import generate_bank, template_examples
from .genai.judge import judge_fidelity
from .genai.narrate import narrate_alerts
from .genai.transcript_bank import generate_bank as build_transcript_bank
from .generate.campaign import generate_dataset
from .generate.entities import build_population
from .generate.fidelity import assess
from .identify.library import load_library, summarise
from .loop.coevolution import run_loop
from .report import RunWriter
from .sweep import KNOBS, across_seeds, sensitivity

# Apple's Accelerate BLAS raises spurious invalid/overflow flags on large dot products.
# The results are correct; the warnings are not, and they bury real output.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def _paths(cfg: Config) -> Tuple[Path, Path, Path]:
    base = cfg.data_path / cfg.run_name
    base.mkdir(parents=True, exist_ok=True)
    return (base / "transactions.parquet", base / "agent_corpus.parquet",
            base / "transcripts.parquet")


def _load_dataset(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    txn_path, corpus_path, transcript_path = _paths(cfg)
    if not txn_path.exists():
        raise SystemExit(
            f"no generated dataset at {txn_path}. Run `python -m redteam generate` first."
        )
    corpus = pd.read_parquet(corpus_path) if corpus_path.exists() else pd.DataFrame()
    # Absent for datasets generated before the transcript plane existed. Defence degrades to
    # its previous behaviour rather than refusing to run against an older cache.
    transcripts = pd.read_parquet(transcript_path) if transcript_path.exists() else pd.DataFrame()
    return pd.read_parquet(txn_path), corpus, transcripts


def _timed(label: str, fn, *args, **kwargs):
    start = time.time()
    print(f"[{label}]")
    out = fn(*args, **kwargs)
    elapsed = time.time() - start
    print(f"[{label}] done in {elapsed:.1f}s\n")
    # Recorded onto whichever writer was passed to the stage, so the report can publish
    # per-stage wall clock rather than the write-up quoting a number somebody remembered.
    for candidate in (*args, *kwargs.values()):
        if isinstance(candidate, RunWriter):
            candidate.record_timing(label, elapsed)
            break
    return out


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------

def stage_identify(cfg: Config, writer: Optional[RunWriter] = None):
    library = load_library()
    problems = library.validate()
    summary = summarise(library)
    print(f"  {summary['total_vectors']} vectors, {summary['simulated_vectors']} simulated, "
          f"{len(summary['families'])} families")
    if problems:
        print("  validation problems:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  library validates against the transaction schema")
    if writer:
        writer.identify(library, problems)
    return library


def stage_ideate(cfg: Config, library, args: argparse.Namespace,
                 writer: Optional[RunWriter] = None):
    """The LLM attack-ideation agent, gated by the library's own validator.

    Separate from ``identify`` rather than folded into it, because ideation proposes and a
    human merges. An agent that edits the taxonomy it is scored against, unsupervised, grows
    a vector count rather than coverage.
    """
    cache = LLMCache(refresh=args.refresh_llm)
    try:
        result = propose_vectors(library, cache, n=args.propose)
    except CacheMiss as exc:
        raise SystemExit(
            f"{exc}\n\nThe generative layer runs from a committed cache so that runs are "
            f"reproducible offline. To generate new completions, set "
            f"$REDTEAM_LLM_API_KEY and pass --refresh-llm."
        ) from exc

    summary = result.summary()
    print(f"  proposed {summary['proposed']}, accepted {summary['accepted']}, "
          f"rejected {summary['rejected']} (accept rate {summary['accept_rate']:.0%})")
    for proposal in result.proposals:
        verdict = "accept" if proposal.accepted else "reject"
        detail = "" if proposal.accepted else f" - {'; '.join(proposal.problems)[:90]}"
        print(f"    [{verdict}] {proposal.vector_id}{detail}")
    if result.accepted:
        print("\n  YAML for review (not merged automatically):\n")
        print("\n".join(f"    {line}" for line in to_yaml_block(result.accepted).splitlines()))
    if writer:
        writer.ideate(result, cache.stats)
    return result


def _payload_bank(args: argparse.Namespace):
    """Load the model-written injection payloads, if any are cached.

    Absent cache means an empty bank and a corpus built exactly as it was before. This is
    the one place a cache miss is not fatal, because the bank improves the injection guard's
    evaluation rather than being required to produce it.
    """
    cache = LLMCache(refresh=args.refresh_llm)
    bank = generate_bank(cache, template_examples())
    if bank:
        s = bank.summary()
        print(f"  payload bank: {s['payloads']} model-written injections across "
              f"{s['families']} families")
    else:
        print("  payload bank: empty, injections are template-composed only")
    return bank


def _transcript_bank(args: argparse.Namespace):
    """Load the model-written call transcripts, if any are cached.

    Same contract as the payload bank: absent cache is not fatal, the transcript corpus falls
    back to template composition, and the run says which it used rather than leaving the
    reader to guess how the vishing guard's numbers were produced.
    """
    cache = LLMCache(refresh=args.refresh_llm)
    bank = build_transcript_bank(cache)
    if bank:
        s = bank.summary()
        print(f"  transcript bank: {s['coercive_transcripts']} model-written scam "
              f"conversations across {s['scripts']} scripts, {s['genuine_transcripts']} genuine")
    else:
        print("  transcript bank: empty, conversations are template-composed only")
    return bank


def stage_generate(cfg: Config, library, writer: Optional[RunWriter] = None,
                   payload_bank=None, transcript_bank=None,
                   args: Optional[argparse.Namespace] = None):
    result = generate_dataset(cfg, library, payload_bank=payload_bank,
                              transcript_bank=transcript_bank)
    txns, corpus, transcripts = result.transactions, result.corpus, result.transcripts
    fidelity = assess(txns)
    # The one fidelity check that cannot be tuned against, because it reads the joint
    # plausibility of a row rather than any marginal. Optional: absent cache means it is
    # simply not reported, rather than the run failing.
    judged = judge_fidelity(txns, LLMCache(refresh=bool(args and args.refresh_llm)),
                            seed=cfg.seed)

    txn_path, corpus_path, transcript_path = _paths(cfg)
    txns.to_parquet(txn_path, index=False)
    corpus.to_parquet(corpus_path, index=False)
    transcripts.to_parquet(transcript_path, index=False)

    print(f"  {len(txns):,} transactions, {int(txns['is_fraud'].sum()):,} fraudulent "
          f"({txns['is_fraud'].mean():.2%}) across "
          f"{txns.loc[txns['is_fraud'] == 1, 'attack_vector_id'].nunique()} vectors")
    print(f"  {len(corpus):,} agent context bundles "
          f"({int(corpus['has_injection'].sum()):,} carrying an injected payload)")
    if not transcripts.empty:
        print(f"  {len(transcripts):,} call and chat transcripts "
              f"({int(transcripts['is_coercive'].sum()):,} coercive)")
    # A component that scored zero has to appear here even when nothing failed outright.
    # "fidelity 0.69, no flags" over a derived-separability score of 0.0 is the report
    # hiding its own worst number, and a full-size run reached exactly that state.
    print(f"  fidelity {fidelity.overall}"
          + (f", flags: {', '.join(fidelity.flags)}" if fidelity.flags else ", no flags")
          + (f"\n  WARNING: {'; '.join(fidelity.warnings)}" if fidelity.warnings else "")
          + (f"\n  scored zero: {', '.join(fidelity.zeroed)}" if fidelity.zeroed else ""))
    if judged:
        print(f"  llm-as-judge: {judged.accuracy:.0%} discriminator accuracy on "
              f"{len(judged.verdicts)} rows (chance is 50%)")
    print(f"  written to {txn_path.parent}")
    if writer:
        writer.generate(txns, corpus, result.per_vector, fidelity, transcripts=transcripts,
                        judged=judged)
    return result


def stage_defend(cfg: Config, txns: pd.DataFrame, corpus: pd.DataFrame,
                 writer: Optional[RunWriter] = None, *, deep: bool = True,
                 transcripts: Optional[pd.DataFrame] = None,
                 args: Optional[argparse.Namespace] = None):
    artifacts = run_defence(txns, corpus, cfg, with_zero_day=deep, with_importance=deep,
                            transcripts=transcripts)
    h = artifacts.headline
    print(f"  recall {h.recall:.1%} | precision {h.precision:.1%} | value recall "
          f"{h.value_recall:.1%} | FPR {h.false_positive_rate:.3%} | PR AUC {h.pr_auc:.4f}")
    worst = artifacts.per_vector.head(3)
    for row in worst.itertuples():
        print(f"  weakest: {row.attack_vector_id} recall {row.recall:.1%} ({row.rows} rows)")

    # Narratives are generated last and cannot touch a decision. Their absence is normal -
    # the cache ships without them unless somebody has run with a key - so a miss prints a
    # note and the run continues.
    narratives = narrate_alerts(artifacts.scored, artifacts.featured,
                                LLMCache(refresh=bool(args and args.refresh_llm)))
    if narratives:
        print(f"  analyst narratives: {len(narratives.narratives)} alerts summarised")

    _save_test_features(cfg, artifacts)

    if writer:
        writer.defend(artifacts, narratives=narratives)
        # Persist the servable artefact and the alert queue. Done here rather than in a
        # separate stage so that the thing the API serves is always the thing the report
        # describes, identified by the same run id.
        _publish(cfg, artifacts, writer.provenance)
    return artifacts


def _save_test_features(cfg: Config, artifacts) -> None:
    """Persist the featured test window so a transaction can be explained after the fact.

    The CSV artefacts carry a transaction's *verdict* - score, decision, reason codes - but
    not the 170-column vector the verdict was computed from. That is enough to report on the
    run and not enough to interrogate a single payment: an inspector showing a field against
    the legitimate baseline needs the field, and a counterfactual needs to re-score a
    perturbed copy of the row through the deployed model, which needs all of them.

    Written to the data directory rather than the artefacts directory because it is bulk
    working data of the same kind as ``transactions.parquet``, not a reportable table, and
    the artefacts directory is the part meant to be read by a human.

    Best-effort, like ``_publish``: a completed run must not fail because a serving
    convenience could not be written.
    """
    try:
        featured = artifacts.featured
        test_ids = set(artifacts.split.test["txn_id"])
        frame = featured[featured["txn_id"].isin(test_ids)]
        # Internal red-team scratch columns are prefixed and are not part of any contract.
        frame = frame[[c for c in frame.columns if not c.startswith("_")]]
        path = cfg.data_path / cfg.run_name / "test_features.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        print(f"  test features not saved: {exc}")


def _publish(cfg: Config, artifacts, provenance) -> None:
    """Write the serving bundle and load the scored test set into SQLite.

    Both are best-effort. A pipeline run whose numbers are already computed should not fail
    because a serving directory was read-only, and the offline path must not acquire a hard
    dependency on the serving layer.
    """
    try:
        from .serve.bundle import from_artifacts
        from .serve.store import Store
    except ImportError:  # pragma: no cover - serving extras absent
        return
    try:
        path = from_artifacts(artifacts, provenance).save(
            cfg.artifacts_path / "serving" / "bundle.pkl")
        store = Store(cfg.data_path / "redteam.sqlite")
        store.record_run(provenance, artifacts.headline.to_dict())
        rows = store.record_alerts(provenance.run_id, artifacts.scored)
        print(f"  published: bundle -> {path}, {rows:,} scored rows -> sqlite "
              f"(run {provenance.run_id})")
    except (OSError, ValueError) as exc:
        print(f"  publish skipped: {exc}")


def stage_loop(cfg: Config, txns: pd.DataFrame, artifacts, writer: Optional[RunWriter] = None,
               library=None):
    # The population is a pure function of the seed, so it is rebuilt here rather than
    # threaded through every stage. The victim-cohort gene needs it to retarget a campaign
    # at a different kind of customer, and without it that gene silently does nothing.
    population = build_population(cfg, np.random.default_rng(cfg.seed))
    result = run_loop(txns, artifacts.featured, artifacts.per_vector, artifacts.detector, cfg,
                      library=library or load_library(),
                      customers=population.customers)
    if not result.rounds:
        print(f"  no rounds completed: no campaign reaches loop.min_fraud_rows "
              f"({cfg.loop.min_fraud_rows}). Generate more days or lower the floor.")
        return result

    def pct(x: float) -> str:
        return "n/a" if x != x else f"{x:+.1%}"

    for r in result.rounds:
        roi = "n/a" if r.attacker_roi != r.attacker_roi else f"{r.attacker_roi:+.2f}"
        print(f"  round {r.round_index}: red took {pct(r.recall_lost_to_red)} from targeted "
              f"vectors ({r.rows_targeted} rows) at ROI {roi}; blue chose "
              f"{r.blue_move} and recovered {pct(r.recall_recovered_by_blue)}, "
              f"FPR {r.post_retrain_fpr:.3%}")
    print(f"  {result.convergence.get('verdict', 'not characterised')}")
    if result.discovered:
        print(f"  {len(result.discovered)} tactics fed back into the taxonomy as new vectors"
              f" ({len(result.discovery_rejections)} rejected by the schema gate)")
    if writer:
        writer.loop(result)
    return result


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redteam",
        description="Closed-loop red team / blue team system for GenAI-era payment fraud.",
    )
    parser.add_argument("stage",
                        choices=["identify", "ideate", "generate", "defend", "loop", "run",
                                 "sweep", "serve"],
                        help="which pillar to run; 'run' does all of them in order, "
                             "'ideate' asks the LLM agent for new attack vectors, "
                             "'sweep' repeats the run across seeds or a parameter, "
                             "'serve' starts the HTTP API over the last run's bundle")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--quick", action="store_true",
                        help="small, fast profile for smoke tests and development")
    parser.add_argument("--seed", type=int, default=None, help="override the run seed")
    parser.add_argument("--run-name", default=None, help="override the run name")
    parser.add_argument("--shallow", action="store_true",
                        help="skip leave-one-vector-out and permutation importance")
    parser.add_argument("--no-loop", action="store_true",
                        help="with 'run', stop after the defence")
    parser.add_argument("--seeds", type=int, default=5, metavar="N",
                        help="with 'sweep', how many seeds to run (default 5)")
    parser.add_argument("--sweep", default=None, metavar="PARAM=V1,V2,...",
                        help="with 'sweep', vary one parameter instead of the seed, "
                             f"e.g. --sweep target_fraud_rate=0.001,0.0075,0.02. "
                             f"Available: {', '.join(sorted(KNOBS))}")
    parser.add_argument("--refresh-llm", action="store_true",
                        help="call the LLM provider for cache misses and write them back; "
                             "needs $REDTEAM_LLM_API_KEY. Without this every completion is "
                             "served from the committed cache and the run is reproducible")
    parser.add_argument("--propose", type=int, default=6, metavar="N",
                        help="with 'ideate', how many vectors to ask for (default 6)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="with 'serve', bind address (default loopback: this API has no "
                             "authentication and should not be exposed without one)")
    parser.add_argument("--port", type=int, default=8000,
                        help="with 'serve', bind port (default 8000)")
    return parser


def _parse_sweep(spec: str) -> Tuple[str, list]:
    if "=" not in spec:
        raise SystemExit(f"--sweep expects PARAM=V1,V2,... (got '{spec}')")
    knob, raw = spec.split("=", 1)
    knob = knob.strip()
    if knob not in KNOBS:
        raise SystemExit(f"unknown sweep parameter '{knob}'. "
                         f"Choose one of: {', '.join(sorted(KNOBS))}")
    try:
        values = [float(v) for v in raw.split(",") if v.strip()]
    except ValueError as exc:
        raise SystemExit(f"--sweep values must be numbers: {exc}") from exc
    if not values:
        raise SystemExit("--sweep needs at least one value")
    return knob, values


def stage_sweep(cfg: Config, args: argparse.Namespace, writer: Optional[RunWriter] = None):
    """Repeated runs, so the headline is reported as an estimate rather than a number."""
    if args.sweep:
        knob, values = _parse_sweep(args.sweep)
        print(f"  sweeping {knob} over {values}")
        result = sensitivity(cfg, knob, values)
    else:
        seeds = [cfg.seed + i for i in range(max(1, args.seeds))]
        print(f"  {len(seeds)} seeds: {seeds}")
        result = across_seeds(cfg, seeds)

    if not result.summary.empty:
        recall = result.summary[result.summary.get("metric", "") == "recall"]
        if not recall.empty:
            r = recall.iloc[0]
            print(f"\n  recall {r['mean']:.1%} +/- {r['sd']:.1%} across {int(r['runs'])} runs "
                  f"(min {r['min']:.1%}, max {r['max']:.1%})")
    if writer:
        writer.sweep(result)
    return result


def load_config(args: argparse.Namespace) -> Config:
    path = args.config or (ROOT / "configs" / "quick.yaml" if args.quick else None)
    cfg = Config.load(path)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_name:
        cfg.run_name = args.run_name
    return cfg


def stage_serve(cfg: Config, args) -> int:
    """Start the HTTP surface over whatever the last defend stage published.

    Split out of the pipeline path entirely: this never generates, never fits and never
    writes a report, so it does not take a RunWriter and does not stamp a run.
    """
    try:
        import uvicorn

        from .serve.api import create_app
    except ImportError as exc:
        raise SystemExit(
            f"the API needs the serving extra: pip install -r requirements-serve.txt ({exc})"
        ) from exc

    bundle = cfg.artifacts_path / "serving" / "bundle.pkl"
    if not bundle.exists():
        print(f"note: no bundle at {bundle}. The API will start and report itself degraded; "
              f"run `python -m redteam run --quick` to build one.")
    print(f"serving on http://{args.host}:{args.port}  (docs at /docs)")
    uvicorn.run(create_app(bundle, cfg.data_path / "redteam.sqlite"),
                host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args)
    if args.stage == "serve":
        return stage_serve(cfg, args)
    writer = RunWriter(cfg)
    deep = not args.shallow

    print(f"run '{cfg.run_name}' | seed {cfg.seed} | "
          f"{cfg.population.n_customers:,} customers x {cfg.benign.n_days} days\n")

    if args.stage == "identify":
        _timed("identify", stage_identify, cfg, writer)
    elif args.stage == "ideate":
        library = stage_identify(cfg)
        _timed("ideate", stage_ideate, cfg, library, args, writer)
    elif args.stage == "generate":
        library = stage_identify(cfg)
        _timed("generate", stage_generate, cfg, library, writer, _payload_bank(args),
               _transcript_bank(args), args=args)
    elif args.stage == "defend":
        txns, corpus, transcripts = _load_dataset(cfg)
        _timed("defend", stage_defend, cfg, txns, corpus, writer, deep=deep,
               transcripts=transcripts, args=args)
    elif args.stage == "loop":
        txns, corpus, transcripts = _load_dataset(cfg)
        artifacts = _timed("defend", stage_defend, cfg, txns, corpus, writer, deep=False,
                           transcripts=transcripts, args=args)
        _timed("loop", stage_loop, cfg, txns, artifacts, writer)
    elif args.stage == "sweep":
        _timed("sweep", stage_sweep, cfg, args, writer)
    else:
        library = _timed("identify", stage_identify, cfg, writer)
        result = _timed("generate", stage_generate, cfg, library, writer,
                        _payload_bank(args), _transcript_bank(args), args=args)
        artifacts = _timed("defend", stage_defend, cfg, result.transactions, result.corpus,
                           writer, deep=deep, transcripts=result.transcripts, args=args)
        if not args.no_loop:
            _timed("loop", stage_loop, cfg, result.transactions, artifacts, writer, library)

    report = writer.write()
    print(f"report written to {report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
