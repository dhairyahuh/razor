# Media

Screenshots of the operator console, and the project card. Everything here is
generated, not mocked up — the route captures are the real application reading the
committed `default_run` run in offline Demo Mode.

| file | what it is |
| --- | --- |
| `card-560x280.png` | Submission card at exact size |
| `card-560x280@2x.png` | The same at 2x, for anywhere that wants a sharper asset |
| `01-live-console.png` | `/` — live payment stream, the argument strip, headline metrics |
| `02-attack-atlas.png` | `/identify` — kill-chain matrix, residual risk |
| `03-fidelity-lab.png` | `/generate` — both separability probes against the deployed band and the ceiling |
| `04-defence.png` | `/defend` — operating curve, per-vector recall with Wilson intervals, baselines |
| `05-arms-race.png` | `/loop` — the rounds, the live round, and the write-back into the taxonomy |
| `06-deployment.png` | `/deploy` — inline decision path, provable blocks, loss bound, cost |
| `07-evidence.png` | `/evidence` — limitations first, then provenance and the run's own report |

## Reproducing them

The route captures need the console running:

```bash
cd frontend && npm install && npm run fonts && npm run dev
```

Then, from a directory with `puppeteer` installed:

```bash
node docs/media/capture.mjs docs/media
```

Captures are taken at 1600×1000 with a device pixel ratio of 2, and with `?mode=demo`
on every URL. Forcing the mode matters: without it each page spends its first 1.2
seconds probing for a backend, and the screenshot races that probe.

## The card

`card.html` is the source. It reads the console's own design tokens and the
self-hosted fonts, which resolve from the dev server's root — so to re-render it,
copy it into `frontend/public/` first, capture `http://localhost:5173/card.html` at
560×280, then remove it again. It lives here rather than in `public/` so a
production build doesn't serve a stray page.

The four figures on the card are the same ones the console shows: 67 vectors from
the taxonomy, 281,906 payments from the full run, 71.6% recall from the honest quick
profile, and 12 arms-race rounds.
