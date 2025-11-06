winget install Ollama.Ollama -s winget
Start-Service Ollama
ollama pull llama3.1:8b

curl http://127.0.0.1:11434/api/tags