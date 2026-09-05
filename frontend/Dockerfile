# The console, built and then served as static files.
#
# Two stages. The first needs Node and 300MB of node_modules; the second needs neither, and
# ships about 6MB of JavaScript, CSS, fonts and the baked demo payload behind nginx. A judge
# pulling this should not be downloading a toolchain to look at a dashboard.

# -- build ---------------------------------------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# Manifests first. Docker caches this layer on their content, so editing a component does not
# reinstall the dependency tree - which is the difference between a five-second rebuild and a
# ninety-second one during development.
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY . .

# Fonts are copied out of the @fontsource packages into public/fonts/ and declared by hand in
# src/styles/fonts.css. They are committed too; this re-runs so a fresh checkout that skipped
# the commit still gets them.
RUN npm run fonts

# Built as a demo bundle: no probe for a backend that may not be deployed, no two-second wait
# for it to fail. Compose overrides this for the paired stack, where the API is up.
ARG VITE_MODE=demo
ENV VITE_MODE=$VITE_MODE
RUN npm run build

# -- serve ---------------------------------------------------------------------------------
FROM nginx:1.27-alpine AS serve

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# The container is useless if nginx is up but the bundle did not copy, which a process check
# would not catch.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://localhost/ >/dev/null || exit 1
