import builtins

from tiktoken import _educational


def test_train_simple_encoding_reads_utf8_source(monkeypatch, capsys):
    def open_with_gbk_default(file, mode="r", **kwargs):
        kwargs.setdefault("encoding", "gbk")
        return builtins.open(file, mode, **kwargs)

    # Emulate a non-UTF-8 locale without changing the process or OS locale.
    monkeypatch.setattr(_educational, "open", open_with_gbk_default, raising=False)

    encoding = _educational.train_simple_encoding()

    tokens = encoding.encode("hello world", visualise=None)
    assert encoding.decode(tokens) == "hello world"
