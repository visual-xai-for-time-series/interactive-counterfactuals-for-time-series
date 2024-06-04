export UID=$(id -u)
export GID=$(id -g)

docker compose build preprocess
docker compose up preprocess
