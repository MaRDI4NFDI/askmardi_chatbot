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
docker run --rm -p 8501:8501 -e LLM_API_KEY=... askmardi-chatbot:dev
```
