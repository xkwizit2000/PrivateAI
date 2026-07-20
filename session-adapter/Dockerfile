FROM oven/bun:1.3-alpine

WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY src ./src
COPY tsconfig.json ./

ENV SESSION_STORAGE_PATH=/data/storage.db
VOLUME ["/data"]

CMD ["bun", "run", "src/index.ts"]
