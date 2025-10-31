# ---- Build React ----
FROM node:20-alpine AS builder
WORKDIR /app

# Only copy package.json (no lock file)
COPY package.json ./
RUN npm ci          # installs exact versions from package.json

COPY . .
RUN npm run build   # creates /app/build

# ---- Serve with Nginx ----
FROM nginx:stable-alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]