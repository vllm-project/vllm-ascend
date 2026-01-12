# vLLM Ascend Plugin documents

Live doc: https://docs.vllm.ai/projects/ascend

## Build the docs

```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html

# Build the docs with translation
make intl

# Open the docs with your browser
python -m http.server -d _build/html/
```

Launch your browser and open:
- English version: http://localhost:8000
- Chinese version: http://localhost:8000/zh_CN
