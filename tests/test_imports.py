"""이식된 v1 모듈이 import 가능한지 확인 (LLM 키·벡터스토어 불필요)."""
import importlib

import pytest

MODULES = [
  "src.config",
  "src.graph.state",
  "src.graph.pipeline",
  "src.vectordb.store",
  "src.ingest.chunker",
  "src.api.api",
  "src.ui.components.uploader",
]

@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
  importlib.import_module(mod)
