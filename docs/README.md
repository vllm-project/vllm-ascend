# vLLM Ascend Plugin documents

Live doc: https://vllm-ascend.readthedocs.io

## Build the docs

```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html

# Extract po files
make po

# Do translations

# Build the docs with translation
make intl

# Open the docs with your browser
python -m http.server -d _build/html/
```

Launch your browser and open:
- English version: http://localhost:8000
- Chinese version: http://localhost:8000/zh_CN
