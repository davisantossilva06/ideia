# instalar Ollama via winget
winget install Ollama.Ollama -s winget

# iniciar serviço
Start-Service Ollama

# puxar modelo
ollama pull llama3.1:8b

# teste rápido
curl http://127.0.0.1:11434/api/tags