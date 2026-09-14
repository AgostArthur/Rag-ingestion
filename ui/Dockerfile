FROM node:22-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY index.html vite.config.js ./
COPY src ./src
ARG VITE_GOOGLE_MAPS_API_KEY=
ENV VITE_GOOGLE_MAPS_API_KEY=$VITE_GOOGLE_MAPS_API_KEY
ARG VITE_GOOGLE_MAPS_MAP_ID=DEMO_MAP_ID
ENV VITE_GOOGLE_MAPS_MAP_ID=$VITE_GOOGLE_MAPS_MAP_ID
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
