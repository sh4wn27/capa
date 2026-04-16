# ── CAPA web frontend ─────────────────────────────────────────────────────────
# Multi-stage build for the Next.js 14 frontend.
#
# Build:
#   docker build -t capa-web ./web
#
# Run (standalone; expects backend at CAPA_BACKEND_URL):
#   docker run -p 3000:3000 \
#     -e CAPA_BACKEND_URL=http://backend:8000 \
#     capa-web
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency install ───────────────────────────────────────────────
FROM node:20-alpine AS deps

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline --no-audit --no-fund


# ── Stage 2: builder ──────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy installed node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build-time env vars (baked into static JS at build time via Next.js NEXT_PUBLIC_*)
ARG NEXT_PUBLIC_APP_ENV=production

RUN npm run build


# ── Stage 3: runtime ──────────────────────────────────────────────────────────
FROM node:20-alpine AS runtime

RUN apk add --no-cache libc6-compat

# Non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser  --system --uid 1001 nextjs

WORKDIR /app

# Copy build output and required files
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs

EXPOSE 3000

ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# CAPA_BACKEND_URL is read at request time by the API route (server-side),
# so it doesn't need to be baked in at build time.
ENV CAPA_BACKEND_URL=""

CMD ["npm", "start"]
