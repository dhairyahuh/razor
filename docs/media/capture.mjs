import fs from 'node:fs'
import path from 'node:path'
import puppeteer from 'puppeteer'

const BASE = 'http://localhost:5173'
const OUT = process.argv[2]
fs.mkdirSync(OUT, { recursive: true })

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

/** Routes for the media gallery, in the order the story should be told. */
const ROUTES = [
  { slug: '01-live-console',   path: '/',         wait: 5200, note: 'live stream + argument strip' },
  { slug: '02-attack-atlas',   path: '/identify', wait: 2600, note: 'kill-chain matrix' },
  { slug: '03-fidelity-lab',   path: '/generate', wait: 2600, note: 'separability vs band and ceiling' },
  { slug: '04-defence',        path: '/defend',   wait: 3200, note: 'operating curve, CIs, baselines' },
  { slug: '05-arms-race',      path: '/loop',     wait: 3200, note: 'rounds, live round, write-back' },
  { slug: '06-deployment',     path: '/deploy',   wait: 2800, note: 'inline path, guards, cost' },
  { slug: '07-evidence',       path: '/evidence', wait: 2800, note: 'limitations first, provenance' },
]

const browser = await puppeteer.launch({
  headless: 'new',
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--force-color-profile=srgb',
         '--font-render-hinting=none', '--hide-scrollbars'],
})

const errors = []

async function shoot(page, url, file, clip = null) {
  await page.goto(url, { waitUntil: 'networkidle2', timeout: 60000 })
  return page
}

/* ---- the card ---- */
{
  const page = await browser.newPage()
  page.on('pageerror', (e) => errors.push(`card: ${e.message}`))

  for (const [scale, name] of [[2, 'card-560x280@2x.png'], [1, 'card-560x280.png']]) {
    await page.setViewport({ width: 560, height: 280, deviceScaleFactor: scale })
    await page.goto(`${BASE}/_card.html`, { waitUntil: 'networkidle2', timeout: 60000 })
    await page.evaluate(() => document.fonts.ready)
    await sleep(600)
    await page.screenshot({ path: path.join(OUT, name), type: 'png' })
    console.log(`  ${name}`)
  }
  await page.close()
}

/* ---- the routes ---- */
{
  const page = await browser.newPage()
  page.on('pageerror', (e) => errors.push(`route: ${e.message}`))
  page.on('console', (m) => {
    if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) {
      errors.push(`console: ${m.text().slice(0, 160)}`)
    }
  })

  // 1600x1000 is comfortably past the console's 1024px floor, and a 16:10 frame
  // matches what a judge will have on a laptop.
  await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 2 })

  for (const r of ROUTES) {
    // `?mode=demo` skips the 1.2s backend probe and makes the capture deterministic,
    // which matters because the probe failing is what a screenshot would otherwise race.
    await page.goto(`${BASE}${r.path}?mode=demo`, { waitUntil: 'networkidle2', timeout: 60000 })
    await page.evaluate(() => document.fonts.ready)
    await sleep(r.wait)
    const file = `${r.slug}.png`
    await page.screenshot({ path: path.join(OUT, file), type: 'png' })
    const kb = (fs.statSync(path.join(OUT, file)).size / 1024).toFixed(0)
    console.log(`  ${file.padEnd(24)} ${String(kb).padStart(5)} KB   ${r.note}`)
  }
  await page.close()
}

await browser.close()

console.log(`\nwrote ${fs.readdirSync(OUT).length} file(s) to ${OUT}`)
if (errors.length) {
  console.log(`\n${errors.length} page error(s):`)
  for (const e of [...new Set(errors)].slice(0, 10)) console.log(`  ! ${e}`)
} else {
  console.log('no page errors')
}
