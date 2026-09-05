/**
 * Tailwind reads the semantic layer of src/styles/tokens.css and nothing else.
 * There is no hex value in this file on purpose: a colour that exists only here
 * is a colour that cannot be themed, and the brief requires both themes to be
 * correct at token level.
 */

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['class', '[data-theme="dark"]'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    // Replacing rather than extending. Inheriting Tailwind's default palette
    // would let `bg-slate-800` compile, which is exactly the escape hatch that
    // makes a theme leak.
    colors: {
      transparent: 'transparent',
      current: 'currentColor',
      ground: 'var(--bg-ground)',
      surface: 'var(--bg-surface)',
      raised: 'var(--bg-raised)',
      overlay: 'var(--bg-overlay)',
      inset: 'var(--bg-inset)',
      hover: 'var(--bg-hover)',
      selected: 'var(--bg-selected)',

      line: {
        subtle: 'var(--border-subtle)',
        DEFAULT: 'var(--border)',
        strong: 'var(--border-strong)',
      },

      fg: {
        DEFAULT: 'var(--fg-primary)',
        secondary: 'var(--fg-secondary)',
        muted: 'var(--fg-muted)',
        faint: 'var(--fg-faint)',
        inverse: 'var(--fg-inverse)',
      },

      accent: {
        DEFAULT: 'var(--accent)',
        strong: 'var(--accent-strong)',
        dim: 'var(--accent-dim)',
        fg: 'var(--accent-fg)',
        wash: 'var(--accent-wash)',
      },

      red: {
        DEFAULT: 'var(--red)',
        strong: 'var(--red-strong)',
        dim: 'var(--red-dim)',
        wash: 'var(--red-wash)',
      },
      amber: {
        DEFAULT: 'var(--amber)',
        strong: 'var(--amber-strong)',
        dim: 'var(--amber-dim)',
        wash: 'var(--amber-wash)',
      },
      green: {
        DEFAULT: 'var(--green)',
        strong: 'var(--green-strong)',
        dim: 'var(--green-dim)',
        wash: 'var(--green-wash)',
      },
      money: {
        DEFAULT: 'var(--money)',
        wash: 'var(--money-wash)',
      },
      grid: 'var(--grid)',
    },

    fontFamily: {
      display: 'var(--font-display)',
      sans: 'var(--font-body)',
      mono: 'var(--font-mono)',
    },

    // Sizes derive from --scale so Presentation Mode moves all of them at once.
    fontSize: {
      '2xs': ['var(--text-2xs)', { lineHeight: '1.35' }],
      xs: ['var(--text-xs)', { lineHeight: '1.4' }],
      sm: ['var(--text-sm)', { lineHeight: '1.45' }],
      base: ['var(--text-base)', { lineHeight: '1.5' }],
      lg: ['var(--text-lg)', { lineHeight: '1.45' }],
      xl: ['var(--text-xl)', { lineHeight: '1.3' }],
      '2xl': ['var(--text-2xl)', { lineHeight: '1.2' }],
      '3xl': ['var(--text-3xl)', { lineHeight: '1.1' }],
    },

    borderRadius: {
      none: '0',
      sm: 'var(--radius-sm)',
      DEFAULT: 'var(--radius)',
      lg: 'var(--radius-lg)',
      full: '9999px',
    },

    extend: {
      boxShadow: { overlay: 'var(--shadow-overlay)' },
      spacing: { 4.5: '1.125rem', 13: '3.25rem', 18: '4.5rem' },
      transitionDuration: { 120: '120ms' },
      keyframes: {
        'row-in': {
          from: { opacity: '0', transform: 'translateY(-6px)' },
          to: { opacity: '1', transform: 'none' },
        },
        'fade-in': { from: { opacity: '0' }, to: { opacity: '1' } },
        'slide-over': {
          from: { transform: 'translateX(16px)', opacity: '0' },
          to: { transform: 'none', opacity: '1' },
        },
        'pulse-dot': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.35' },
        },
      },
      animation: {
        'row-in': 'row-in 180ms ease-out',
        'fade-in': 'fade-in 140ms ease-out',
        'slide-over': 'slide-over 180ms cubic-bezier(0.16, 1, 0.3, 1)',
        'pulse-dot': 'pulse-dot 1.8s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
