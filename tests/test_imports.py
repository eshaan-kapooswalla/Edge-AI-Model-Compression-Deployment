from importlib import import_module


def test_imports():
    import_module("src.model_compression.train")
    import_module("src.model_compression.datasets")
    import_module("src.model_compression.models")
