# vLLM Ascend Plugin documents

Live doc: https://vllm-ascend.readthedocs.io

## Build the docs

```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html
```

## Open the docs with your browser

```bash
python -m http.server -d build/html/
```

Launch your browser and open http://localhost:8000/.

