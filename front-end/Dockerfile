# syntax=docker/dockerfile:1
FROM node:14-buster-slim
# In case we want production performance boost
# ENV NODE_ENV=production
WORKDIR /app
COPY ["/package.json", "/package-lock.json*", "./"]
# In case we want production performance boost
# RUN npm install --production
RUN npm install
COPY . .
EXPOSE 4200
# Consider pre-building Angular app
# RUN npm run build
# CMD npm run start
CMD ["npm", "run", "docker"]