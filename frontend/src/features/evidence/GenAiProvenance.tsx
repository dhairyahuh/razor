import { useSummary } from '@/api/queries'

/**
 * Where the language model is, and where it is not.
 *
 * This panel exists because the question is certain to be asked and the answer
 * is better given than extracted. The challenge is about GenAI-enabled fraud, and
 * a reader who never finds a language model at runtime is entitled to wonder
 * whether the "GenAI" is decoration.
 *
 * The answer has two halves and both belong on screen:
 *
 *  1. **The GenAI in the threat model is what the attacks represent**, not a
 *     runtime dependency of the simulator. Voice cloning, deepfake video,
 *     LLM-written pretexts and prompt injection are the *modelled* capability;
 *     simulating a voice-cloned call does not require cloning a voice, it
 *     requires modelling what such a call leaves behind in a payment stream.
 *  2. **Six components genuinely do call a model**, all through a
 *     content-addressed cache. The committed cache is empty, so each one falls
 *     back to a template-composed path — and says so here, in the run log, and in
 *     the report.
 *
 * Populating the cache would need an API key. The alternative — hand-writing
 * completions and committing them as a model's output — would make every claim
 * built on top of them false, which is why the cache ships empty rather than
 * ships fabricated.
 */

interface Component {
  module: string
  does: string
  fallback: string
}

const COMPONENTS: Component[] = [
  {
    module: 'genai/ideate.py',
    does: 'proposes new attack vectors in the library’s own schema',
    fallback: 'the hand-written taxonomy, unchanged',
  },
  {
    module: 'genai/payloads.py',
    does: 'writes prompt-injection payloads for agent context bundles',
    fallback: 'composed from framings, actions, concealments and lexical jitter',
  },
  {
    module: 'genai/transcript_bank.py',
    does: 'writes call and chat transcripts in a second voice',
    fallback: 'template-composed scripts — one author for both classes',
  },
  {
    module: 'genai/agents.py',
    does: 'red and blue agents reasoning inside the loop',
    fallback: 'the genome search and blue’s costed move selection',
  },
  {
    module: 'genai/judge.py',
    does: 'LLM-as-judge scoring generated rows against described real ones',
    fallback: 'not scored — the fidelity composite omits it',
  },
  {
    module: 'genai/narrate.py',
    does: 'turns reason codes into analyst-readable summaries',
    fallback: 'the reason codes themselves',
  },
]

export function GenAiProvenance() {
  const { data } = useSummary()
  const generate = ((data?.summary ?? {}) as Record<string, any>).generate ?? {}

  // The judge is the one component whose absence the run records directly, so it
  // is the honest signal for whether the cache was populated.
  const judged = generate.llm_judge !== null && generate.llm_judge !== undefined

  return (
    <div className="space-y-3 p-3 text-2xs leading-relaxed text-fg-secondary">
      <div
        className={`rounded border p-2 ${
          judged ? 'border-green-dim bg-green-wash' : 'border-amber-dim bg-amber-wash'
        }`}
      >
        <p className={judged ? 'text-green' : 'text-amber'}>
          {judged
            ? 'The LLM cache was populated for this run — model-authored text is in the data.'
            : 'The LLM cache is empty for this run. Every component below ran its template-composed fallback.'}
        </p>
        <p className="mt-1 text-fg-muted">
          A populated cache makes a run byte-identical offline, forever, with no API key — the
          key is a build-time dependency, like a lockfile. Committing hand-written completions
          as a model&apos;s output would make every claim built on them false, so it ships empty
          rather than fabricated.
        </p>
      </div>

      <div>
        <p className="font-medium text-fg">The GenAI is what the attacks are, not what runs them.</p>
        <p className="mt-0.5 text-fg-muted">
          Voice cloning, deepfake video calls, LLM-written pretexts, indirect prompt injection
          and MCP tool poisoning are the modelled capability — 93 distinct GenAI enablers across
          the taxonomy. Simulating a voice-cloned call does not require cloning a voice; it
          requires modelling what one leaves behind in a payment stream. No detection metric on
          this site changes either way, and the report says so.
        </p>
      </div>

      <div>
        <p className="mb-1 font-medium text-fg">What each component does, and what it fell back to</p>
        <ul className="space-y-1.5">
          {COMPONENTS.map((c) => (
            <li key={c.module} className="border-l border-line-subtle pl-2">
              <div className="font-mono text-fg-secondary">{c.module}</div>
              <div className="text-fg-muted">{c.does}</div>
              <div className="text-fg-faint">
                {judged ? 'model-authored' : <>fallback: {c.fallback}</>}
              </div>
            </li>
          ))}
        </ul>
      </div>

      <p className="text-fg-faint">
        What an empty cache costs is concentrated in one place: the vishing guard scores near 1.0
        on every holdout because one author wrote both classes, and no holdout in this repository
        can settle that. Only text from a second author can, which is exactly what the transcript
        bank is for.
      </p>
    </div>
  )
}
