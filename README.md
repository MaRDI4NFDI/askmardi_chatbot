This is the first ASK_MARDI chatbot prototype.

## Running

### Locally
```
python -m streamlit run app/ui.py
```

### Docker

#### Building
```
docker build -f docker/Dockerfile -t askmardi-chatbot:dev .
```

#### Run
```
docker run --rm \
  -p 8501:8501 \
  -e OLLAMA_API_KEY=API_KEY \
  -e QDRANT_URL=http://your-qdrant.com:6333 \
  askmardi-chatbot:dev
```
