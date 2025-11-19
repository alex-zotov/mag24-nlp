# install

ollama
localhost:11434

/etc/hosts
127.0.0.1 host.docker.internal

curl http://host.docker.internal:11434

# load model
ollama pull mistral

# run model
ollama run mistral

# n8n
docker volume create n8n_data

# win macos
docker run -it --rm --add-host host.docker.internal:host-gateway --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n

# linux
docker run -it --rm --network=host --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
