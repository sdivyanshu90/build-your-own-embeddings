# Contributing

Use Python 3.11+, create a virtual environment, and install `.[dev,faiss]`. Add the
smallest behavior-oriented test first, keep standard tests CPU-only and network-free,
then run `make lint`, `make typecheck`, and `make test`. Do not commit generated
artifacts, credentials, private training text, or benchmark claims without the data and
hardware needed to reproduce them.

