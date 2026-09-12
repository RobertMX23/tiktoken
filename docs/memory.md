# Encoding memory usage

`tiktoken.get_encoding` caches each loaded `Encoding` by name for the lifetime of the process.
`tiktoken.encoding_for_model` uses the same cache: models mapped to the same encoding share
one object. Reuse that object for repeated tokenisation rather than constructing an encoding
for every request.

The cache holds a strong reference, so deleting your variable or calling `gc.collect()` does
not unload the encoding. A saved bound method, such as `encode = enc.encode`, also keeps its
encoding alive. There is currently no public API to unload a cached encoding.

`TIKTOKEN_CACHE_DIR` controls the on-disk cache of downloaded vocabulary files. Changing it,
deleting those files, or setting it to an empty string does not clear the in-memory encoding
cache. The empty string disables disk caching, so a later process may need to download the
vocabulary again.

If an application needs tokenizer memory only for one phase of its work, run that phase in a
separate process and let the worker exit when it finishes. The operating system then reclaims
the worker's memory. Load the encoding inside the worker and return token IDs or counts to
the parent; returning an `Encoding` can load it into the parent's cache when it is unpickled.

Save this example as a Python file and run it as a script:

```python
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context


def count_tokens(texts: list[str]) -> list[int]:
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    return [len(enc.encode(text)) for text in texts]


if __name__ == "__main__":
    # A fresh interpreter avoids inheriting an already loaded encoding from the parent.
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        counts = pool.submit(count_tokens, ["hello world", "goodbye world"]).result()
    # Leaving the context waits for the worker to exit; only its results remain here.
    print(counts)
```

Process startup and loading an encoding have a cost. Submit a batch of work to amortise that
cost; a persistent pool retains each worker's encoding until the worker exits. This does not
unload any encodings already loaded in the parent process, and memory used by returned results
remains with the parent.
