# Stage 1: Build Vite app
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# Vite bakes VITE_* vars into the JS bundle at build time.
# These are passed in via docker-compose build.args (sourced from root .env).
ARG VITE_API_URL=""
ARG VITE_OPENAI_KEY=""
ARG VITE_ANTHROPIC_KEY=""
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_OPENAI_KEY=$VITE_OPENAI_KEY
ENV VITE_ANTHROPIC_KEY=$VITE_ANTHROPIC_KEY

RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
