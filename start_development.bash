export UID=$(id -u)
export GID=$(id -g)

docker compose build frontend backend
docker compose up frontend backend
