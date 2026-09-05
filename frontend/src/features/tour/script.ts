/**
 * The four-minute path, as data.
 *
 * The tour is not a feature tour. A judge who clicks nothing must still receive
 * the entire argument, which means these captions are the argument - not labels
 * for what is on screen, but the claim each screen supports and why it should be
 * believed. A caption reading "kill-chain matrix" teaches nothing. A caption
 * reading "forty-six vectors, and the shading is measured recall, not intent"
 * teaches the whole pillar.
 *
 * Written as a table so the path can be read, argued with and re-timed in one
 * place rather than reconstructed from component code. The timings match the
 * judging script: console, inspector, atlas, fidelity, the round, the write-back,
 * the limitations.
 */

export interface Step {
  /** Route to be on. The tour navigates if it is not already there. */
  route: string
  /** `id` of the element to spotlight. Omit to speak over the whole screen. */
  target?: string
  /** Two or three sentences. This is the argument, not a label. */
  caption: string
  /** How long the step holds before advancing, in milliseconds. */
  hold: number
  /**
   * Run something when the step opens - open the inspector, fire a round. The
   * tour pauses on these until they resolve or time out, because the point of
   * the step is the thing happening rather than the caption over it.
   */
  action?: 'open-inspector' | 'run-round'
}

export const SCRIPT: Step[] = [
  {
    route: '/',
    target: 'tour-stream',
    caption:
      'Payments arriving and being scored. Every row here was scored by the trained ensemble ' +
      'during the run — the arrival is replayed, the decision is not. The two kinds of stop ' +
      'are kept visually distinct because they are different products: an amber alert is a ' +
      'probability, a red block is a deterministic guard that proved its case and can be ' +
      'disputed on the facts.',
    hold: 12000,
  },
  {
    route: '/',
    target: 'tour-counters',
    caption:
      'The counters tally what has gone past — value at risk stopped, and the realised ' +
      'false-positive rate against the budget the threshold was calibrated to. They are a ' +
      'running total of what you have watched, not a second measurement, and the panel says ' +
      'so rather than letting two numbers on one screen quietly disagree.',
    hold: 9000,
  },
  {
    route: '/',
    action: 'open-inspector',
    caption:
      'One alerted payment, opened. Reason codes in English, every model input against what ' +
      'that field looks like on legitimate traffic, and — where the pipeline computed one — ' +
      'the single change that would have cleared it. For an agent-initiated payment the ' +
      'context window the agent read is here too, with the injected instruction highlighted ' +
      'at the exact offsets the generator recorded when it wrote them in.',
    hold: 16000,
  },
  {
    route: '/identify',
    target: 'tour-killchain',
    caption:
      'The taxonomy against the stages of a payment attack. The shading is measured recall ' +
      'joined from the defence — not intent, not coverage claimed. An unshaded square is a ' +
      'vector this run never simulated, which is a different and more honest thing to admit ' +
      'than a low score. The residual-risk table below ranks exactly that gap.',
    hold: 14000,
  },
  {
    route: '/generate',
    target: 'tour-separability',
    caption:
      'The question every synthetic-data submission has to answer: did the generator stamp ' +
      'the label into the data? Two probes ask how much fraud a learner recovers from the ' +
      'data alone, before the defence exists. Deployed systems sit at 50–85%; past 92% the ' +
      'pipeline flags its own data as measuring the generator rather than a defence. On this ' +
      'run it fired — which is why the detection numbers here are quoted as an upper bound, ' +
      'and why the smaller profile is the honest one to read.',
    hold: 15000,
  },
  {
    route: '/loop',
    target: 'tour-armsrace',
    caption:
      'The arms race. Three lines per round: recall before red moved, after red’s best genome ' +
      'landed, and after blue retrained in response. The gap between the first two is what the ' +
      'attack cost; the gap between the second and third is what the defence won back. Those ' +
      'two quantities are the loop.',
    hold: 13000,
  },
  {
    route: '/loop',
    target: 'tour-runround',
    action: 'run-round',
    caption:
      'Running it now, live. A campaign is generated for this vector, the evasion levers are ' +
      'applied, and both versions are scored through the deployed model — so the difference ' +
      'between the two numbers is the levers, not sampling. This is red’s half of a round; ' +
      'blue’s half is minutes of retraining and is in the recorded rounds above.',
    hold: 18000,
  },
  {
    route: '/loop',
    target: 'tour-writeback',
    caption:
      'And this is the arrow closing. A mutation that evades, survives validation against the ' +
      'library’s schema, and is written back as a first-class vector — which the next run ' +
      'simulates and measures from the start. The system that was attacked is now looking for ' +
      'the attack. That is the closed loop, and it is the whole submission.',
    hold: 14000,
  },
  {
    route: '/deploy',
    target: 'tour-deploy',
    caption:
      'And whether any of it could run. Two of the three layers sit inside the authorisation ' +
      'window; everything that trains is off the payment path entirely. The probe measures a ' +
      'real decision against the deployed model rather than quoting a batch timing as if it ' +
      'were a latency — and it reports feature completeness from a bare payment message, ' +
      'because most of this model is history a message does not carry.',
    hold: 14000,
  },
  {
    route: '/evidence',
    caption:
      'What these numbers cannot tell you, first rather than last. The data is synthetic, ' +
      'several per-vector recalls rest on a handful of rows, and every rupee figure rests on ' +
      'stated cost assumptions. Naming the limits is what makes the rest of it worth ' +
      'believing — and every figure on every screen links to the artefact it came from, so ' +
      'none of it has to be taken on trust.',
    hold: 15000,
  },
]

/** Roughly four minutes, which is the point. */
export const TOTAL_MS = SCRIPT.reduce((sum, s) => sum + s.hold, 0)
