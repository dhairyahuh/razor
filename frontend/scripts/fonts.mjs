/**
 * Copy the exact font files we use out of the @fontsource packages and into
 * public/fonts/, where src/styles/fonts.css declares them by hand.
 *
 * Why not just import the @fontsource CSS? Because that pulls in every weight
 * and every unicode subset the package ships - about forty files - and the
 * brief is explicit that fonts are self-hosted and accounted for. Six files we
 * chose and can name beats forty we inherited.
 *
 * The output is committed. A checkout without node_modules must still render.
 */

import { copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const modules = join(here, '..', 'node_modules')
const out = join(here, '..', 'public', 'fonts')

/** [source under node_modules, name in public/fonts] */
const FACES = [
  ['@fontsource-variable/archivo/files/archivo-latin-wght-normal.woff2', 'archivo-latin.woff2'],
  ['@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-400-normal.woff2', 'plex-sans-400-latin.woff2'],
  ['@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-500-normal.woff2', 'plex-sans-500-latin.woff2'],
  ['@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-600-normal.woff2', 'plex-sans-600-latin.woff2'],
  ['@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2', 'plex-mono-400-latin.woff2'],
  ['@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-500-normal.woff2', 'plex-mono-500-latin.woff2'],
  ['@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-600-normal.woff2', 'plex-mono-600-latin.woff2'],
]

mkdirSync(out, { recursive: true })

let copied = 0
const missing = []

for (const [from, to] of FACES) {
  const source = join(modules, from)
  if (!existsSync(source)) {
    missing.push(from)
    continue
  }
  copyFileSync(source, join(out, to))
  copied += 1
}

if (missing.length) {
  console.error(`fonts: ${missing.length} source file(s) not found under node_modules:`)
  for (const m of missing) console.error(`  ${m}`)
  console.error('run `npm install` first')
  process.exit(1)
}

console.log(`fonts: copied ${copied} face(s) into public/fonts/`)
