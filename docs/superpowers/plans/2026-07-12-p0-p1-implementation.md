# aigo v2 P0+P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v1(RAG Q&A 챗봇)을 개인 레포로 이식하고(P0), 계약서 PDF → 조항 분해 → 조항별 근거 매칭까지의 탐지 MVP + 파일럿 골든셋 30건(P1)을 구축한다.

**Architecture:** v1의 LangGraph Q&A 파이프라인·Qdrant 리트리벌·결정론적 인용을 그대로 이식하고, 그 옆에 탐지 파이프라인(PDF 추출 → 조항 분해 → LLM 분류 → 근거 리트리벌 → 발견 유형 판정)을 새로 얹는다. 탐지와 Q&A는 `QdrantStore`·인용 로직을 공유한다.

**Tech Stack:** Python 3.12, uv, LangGraph, langchain-openai, Qdrant(로컬 파일 모드), sentence-transformers(kure-v1), PyMuPDF, pydantic, pytest, GitHub Actions.

## Global Constraints

- Python 3.12, 패키지 관리는 uv (`uv sync`, `uv run`).
- **인덴트 2칸** (v1 컨벤션 유지 — PEP8 4칸 아님).
- 커밋 메시지: `<type>: <한국어 설명>`, type ∈ feat|fix|docs|refactor|chore|style|perf|ci|test. 어트리뷰션 푸터 없음.
- **근거제시형 원칙**: 프롬프트·UI 문구·변수명에 "위험/안전 판정" 표현 금지. 시스템 출력은 발견 유형 4종(법령과 다름/분쟁 전례 있음/표준계약서와 상이/특이사항 없음) + 근거 ID만.
- v1은 MIT (Copyright (c) 2026 SKN24-Chatbot) — LICENSE에 원 저작권 고지 유지, README에 팀 크레딧 명기.
- 골든셋에 AI Hub 유래 데이터 포함 금지. 골든셋 라벨은 근거 문서 ID 없이 부여 금지.
- LLM 호출·벡터스토어 필요 테스트는 `@pytest.mark.integration`, CI는 `-m "not integration"`만 실행.
- v1 소스 참조는 `/Users/jpaper/Documents/projects/skn/aigo/aigo-ai`에서 `git show origin/main:<path>` (작업 트리 비어 있음).

---

### Task 1 (P0): 프로젝트 스캐폴드 + CI

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `LICENSE`, `README.md`, `tests/test_sanity.py`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `uv run pytest -m "not integration"`이 통과하는 빈 프로젝트. 이후 모든 태스크가 이 위에서 작업.

- [ ] **Step 1: pyproject.toml 작성**

```toml
[project]
name = "aigo-v2"
version = "0.1.0"
description = "임대차 계약서 위험 조항 탐지 시스템 (근거제시형)"
requires-python = ">=3.12"
dependencies = [
  "langgraph>=0.2",
  "langchain-openai>=0.2",
  "langchain-core>=0.3",
  "qdrant-client>=1.9",
  "sentence-transformers>=3.0",
  "pymupdf>=1.24",
  "pydantic>=2.7",
  "python-dotenv>=1.0",
  "streamlit>=1.35",
  "requests>=2.31",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["integration: LLM 또는 벡터스토어가 필요한 테스트"]
```

- [ ] **Step 2: .gitignore / .env.example / LICENSE / README 작성**

`.gitignore`: v1 것을 복사 후 `db/`, `reports/`, `.env` 포함 확인 (`git show origin/main:.gitignore`).

`.env.example`:
```bash
LLM_API_KEY=
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=
OC=            # 국가법령정보 Open API 키
QDRANT_URL=    # 비우면 로컬 파일 모드(QDRANT_PATH)
QDRANT_PATH=./db
```

`LICENSE`: MIT 전문. 저작권 줄 두 개 유지:
```
Copyright (c) 2026 SKN24-Chatbot
Copyright (c) 2026 임정희
```

`README.md` 최소 골격 (프로젝트 한 줄 소개, "법률 자문이 아닌 정보 제공" 고지, v1 크레딧 문단: SKN 캠프 24기 3차 팀 프로젝트 aigo-ai 기반 개인 재설계임을 명기, 팀원 GitHub 링크는 v1 README 참조).

- [ ] **Step 3: 실패하는 sanity 테스트 작성**

`tests/test_sanity.py`:
```python
def test_sanity():
  assert 1 + 1 == 2
```

- [ ] **Step 4: uv sync 후 테스트 실행**

Run: `uv sync && uv run pytest -v`
Expected: `test_sanity PASS` (1 passed)

- [ ] **Step 5: CI 워크플로 작성**

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { python-version: "3.12" }
      - run: uv sync
      - run: uv run pytest -m "not integration" -v
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: 프로젝트 스캐폴드 및 CI 구성"
```

---

### Task 2 (P0): v1 소스 이식

**Files:**
- Create: `src/` 전체, `scripts/build_vectorstore.py`, `scripts/sync_data.py`, `data/processed/*.jsonl`, `docs/v1/` (prompt_strategy.md, vectordb_usage.md, api.md), `tests/test_imports.py`

**Interfaces:**
- Produces: v1과 동일한 공개 인터페이스 — `src.graph.pipeline.run_preformat(query)/stream_formatter(state)/run(query)`, `src.graph.state.State/Citation`, `src.vectordb.QdrantStore.search(query, top_k, filters)`, `src.vectordb.Embedder`, `src.llm.client.llm/streaming_llm`, `src.config`의 `COLLECTION_NAME/EMBEDDING_MODEL/RETRIEVAL_TOP_K/CONTEXT_TOP_K/RELEVANCE_THRESHOLD`. Task 5~10이 이것들을 소비한다.

- [ ] **Step 1: v1 파일 복사**

```bash
V1=/Users/jpaper/Documents/projects/skn/aigo/aigo-ai
cd /Users/jpaper/Documents/projects/aigo-v2
for f in $(cd $V1 && git ls-tree -r origin/main --name-only | grep -E '^(src/|scripts/|data/processed/)'); do
  mkdir -p "$(dirname "$f")"
  (cd $V1 && git show "origin/main:$f") > "$f"
done
mkdir -p docs/v1
for d in prompt_strategy vectordb_usage api; do
  (cd $V1 && git show "origin/main:docs/$d.md") > "docs/v1/$d.md"
done
```

- [ ] **Step 2: 실패하는 import 스모크 테스트 작성**

`tests/test_imports.py`:
```python
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
]

@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
  importlib.import_module(mod)
```

- [ ] **Step 3: 테스트 실행, import 오류 수정**

Run: `uv run pytest tests/test_imports.py -v`
Expected: 처음엔 의존성/경로 오류로 FAIL 가능. 오류 나는 의존성은 pyproject에 추가(`uv add <pkg>`)하되 v1 `git show origin/main:pyproject.toml`의 버전을 참조. 전부 PASS 될 때까지 수정. **v1 코드 로직은 수정하지 않는다** — import 경로/의존성만 손본다.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: v1 aigo-ai 파이프라인 이식 (MIT, SKN24-Chatbot 크레딧)"
```

---

### Task 3 (P0): 벡터스토어 구축 + Q&A 스모크 (integration)

**Files:**
- Create: `tests/test_qa_smoke.py`

**Interfaces:**
- Produces: 로컬 `./db`에 구축된 Qdrant 컬렉션. Task 9·10의 리트리벌이 이것을 사용.

- [ ] **Step 1: 벡터스토어 스모크 구축**

Run: `uv run python scripts/build_vectorstore.py --limit 100`
Expected: 오류 없이 완료, `./db` 생성. (전체 구축은 `--reset` 없이 재실행)

- [ ] **Step 2: integration 스모크 테스트 작성**

`tests/test_qa_smoke.py`:
```python
import pytest

@pytest.mark.integration
def test_qa_pipeline_end_to_end():
  """Q&A 그래프가 질의에 대해 fallback 또는 인용 포함 답변을 반환한다."""
  from src.graph.pipeline import run

  state = run("보증금 반환이 늦어지면 어떻게 되나요?")
  assert state.get("is_terminated") or state.get("final_answer") or state.get("fallback_message")
```

- [ ] **Step 3: 실행 (`.env`에 LLM_API_KEY 필요)**

Run: `uv run pytest -m integration -v`
Expected: PASS. 실패 시 v1 로직이 아니라 env/데이터 문제부터 의심.

- [ ] **Step 4: 전체 벡터스토어 구축 후 Commit**

```bash
uv run python scripts/build_vectorstore.py
git add -A && git commit -m "test: Q&A 파이프라인 이식 스모크 테스트"
```

---

### Task 4 (P1): 위험 카테고리 taxonomy

**Files:**
- Create: `docs/taxonomy.md`, `src/detect/__init__.py`, `src/detect/taxonomy.py`, `tests/detect/test_taxonomy.py`

**Interfaces:**
- Produces: `Category(str, Enum)` (영문 id), `CATEGORY_LABELS: dict[Category, str]` (한글 라벨), `FindingType(str, Enum)`. Task 5·8·9·10이 소비.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/detect/test_taxonomy.py`:
```python
from src.detect.taxonomy import CATEGORY_LABELS, Category, FindingType


def test_categories_have_labels():
  assert set(CATEGORY_LABELS) == set(Category)
  assert 8 <= len(Category) <= 12


def test_finding_types():
  assert {f.value for f in FindingType} == {
    "법령과 다름", "분쟁 전례 있음", "표준계약서와 상이", "특이사항 없음",
  }
```

- [ ] **Step 2: 실행 → FAIL 확인**

Run: `uv run pytest tests/detect/test_taxonomy.py -v` — Expected: ModuleNotFoundError

- [ ] **Step 3: 구현**

`src/detect/taxonomy.py`:
```python
"""위험 카테고리 및 발견 유형 정의. 근거: docs/taxonomy.md."""
from enum import Enum


class Category(str, Enum):
  DEPOSIT_RETURN = "deposit_return"          # 보증금 반환 방해
  RESTORATION = "restoration"                # 원상복구 과잉 부담
  REPAIR_SHIFT = "repair_shift"              # 수선의무 전가
  EARLY_TERMINATION = "early_termination"    # 중도해지 불리
  COST_SHIFT = "cost_shift"                  # 관리비·비용 전가
  RENEWAL_LIMIT = "renewal_limit"            # 계약갱신 제한
  COLLATERAL = "collateral"                  # 담보·전세권 관련
  RENT_RAISE = "rent_raise"                  # 차임 증액 관련
  NONE = "none"                              # 해당 카테고리 없음


CATEGORY_LABELS: dict[Category, str] = {
  Category.DEPOSIT_RETURN: "보증금 반환 방해",
  Category.RESTORATION: "원상복구 과잉 부담",
  Category.REPAIR_SHIFT: "수선의무 전가",
  Category.EARLY_TERMINATION: "중도해지 불리",
  Category.COST_SHIFT: "관리비·비용 전가",
  Category.RENEWAL_LIMIT: "계약갱신 제한",
  Category.COLLATERAL: "담보·전세권 관련",
  Category.RENT_RAISE: "차임 증액 관련",
  Category.NONE: "해당 없음",
}


class FindingType(str, Enum):
  LAW_CONFLICT = "법령과 다름"
  DISPUTE_PRECEDENT = "분쟁 전례 있음"
  STANDARD_DIFF = "표준계약서와 상이"
  NONE = "특이사항 없음"
```

(테스트의 8~12 범위에 NONE 포함 9개로 충족. 카테고리 조정 시 docs/taxonomy.md와 동기화.)

- [ ] **Step 4: docs/taxonomy.md 작성**

각 카테고리마다: 정의 1문장, 대표 특약 예시 1개, 근거 소스(HUG 전세사기예방센터 유형 / 주택임대차분쟁조정사례집 빈출 유형 / 주택임대차보호법 관련 조문 — 예: RENT_RAISE ↔ 주임법 제7조 증액 상한). 작성 중 카테고리 추가·삭제가 필요하면 enum과 테스트 범위를 함께 수정.

- [ ] **Step 5: 실행 → PASS, Commit**

```bash
uv run pytest tests/detect/test_taxonomy.py -v
git add -A && git commit -m "feat: 위험 카테고리 taxonomy 정의"
```

---

### Task 5 (P1): 골든셋 스키마 + 검증기

**Files:**
- Create: `src/detect/golden.py`, `scripts/validate_golden.py`, `data/golden/pilot.jsonl`, `docs/labeling-guide.md`, `tests/detect/test_golden.py`

**Interfaces:**
- Produces: `GoldenClause` pydantic 모델, `load_golden(path) -> list[GoldenClause]`. Task 10(평가)이 소비.
- Consumes: Task 4의 `Category`, `FindingType`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/detect/test_golden.py`:
```python
import pytest
from pydantic import ValidationError

from src.detect.golden import GoldenClause, load_golden

VALID = {
  "id": "g-001",
  "text": "임차인은 계약 기간 중 어떠한 사유로도 보증금 반환을 요구할 수 없다.",
  "category": "deposit_return",
  "finding_types": ["법령과 다름"],
  "evidence": [{"doc_type": "법령", "source_id": "주택임대차보호법", "detail": "제10조"}],
  "source": "synthetic-test",
}


def test_valid_clause_parses():
  c = GoldenClause.model_validate(VALID)
  assert c.category.value == "deposit_return"


def test_finding_without_evidence_rejected():
  bad = {**VALID, "evidence": []}
  with pytest.raises(ValidationError):
    GoldenClause.model_validate(bad)


def test_none_finding_needs_no_evidence():
  ok = {**VALID, "finding_types": ["특이사항 없음"], "evidence": []}
  GoldenClause.model_validate(ok)


def test_pilot_file_valid():
  clauses = load_golden("data/golden/pilot.jsonl")
  assert len(clauses) >= 5
  assert len({c.id for c in clauses}) == len(clauses)
```

- [ ] **Step 2: 실행 → FAIL 확인**

Run: `uv run pytest tests/detect/test_golden.py -v` — Expected: ModuleNotFoundError

- [ ] **Step 3: 구현**

`src/detect/golden.py`:
```python
"""골든셋 스키마. 규칙: 근거 ID 없는 라벨 금지 (스펙 §7)."""
import json
from pathlib import Path

from pydantic import BaseModel, model_validator

from src.detect.taxonomy import Category, FindingType


class Evidence(BaseModel):
  doc_type: str    # "법령" | "판례" | "법령해석례" | "조정사례"
  source_id: str   # 법령명/법령ID, 판례번호, 사례집 사례번호
  detail: str = "" # 조문번호 등


class GoldenClause(BaseModel):
  id: str
  text: str
  category: Category
  finding_types: list[FindingType]
  evidence: list[Evidence]
  source: str      # 출처 (사례집 페이지, HUG 유형 등)
  notes: str = ""

  @model_validator(mode="after")
  def evidence_required_unless_none(self):
    non_none = [f for f in self.finding_types if f != FindingType.NONE]
    if non_none and not self.evidence:
      raise ValueError("발견 유형이 있으면 근거(evidence)가 필수입니다")
    return self


def load_golden(path: str | Path) -> list[GoldenClause]:
  rows = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(GoldenClause.model_validate(json.loads(line)))
  return rows
```

`scripts/validate_golden.py`:
```python
"""골든셋 검증 CLI: uv run python scripts/validate_golden.py data/golden/pilot.jsonl"""
import sys

from src.detect.golden import load_golden

clauses = load_golden(sys.argv[1])
print(f"OK: {len(clauses)}건 유효")
```

- [ ] **Step 4: 시드 골든셋 5건 작성**

`data/golden/pilot.jsonl` — 아래 5건으로 시작 (형식 예시이자 실제 시드. 라벨 근거는 작성 시 국가법령정보에서 조문 확인 후 확정):
```jsonl
{"id": "g-001", "text": "임대인은 계약 기간 중 차임을 시세에 따라 조정할 수 있으며 임차인은 이에 따른다.", "category": "rent_raise", "finding_types": ["법령과 다름"], "evidence": [{"doc_type": "법령", "source_id": "주택임대차보호법", "detail": "제7조"}], "source": "seed: 주임법 제7조 증액 상한(5%) 상충 유형"}
{"id": "g-002", "text": "임차인은 퇴거 시 도배·장판을 새것으로 교체하여 원상복구한다.", "category": "restoration", "finding_types": ["표준계약서와 상이"], "evidence": [{"doc_type": "조정사례", "source_id": "주택임대차분쟁조정사례집(2022)", "detail": "원상복구 분쟁 사례 — 추출 시 사례번호 기입"}], "source": "seed: 통상 손모 초과 원상복구 유형"}
{"id": "g-003", "text": "보일러 등 시설물 수리는 원인과 무관하게 임차인이 부담한다.", "category": "repair_shift", "finding_types": ["법령과 다름", "분쟁 전례 있음"], "evidence": [{"doc_type": "법령", "source_id": "민법", "detail": "제623조"}, {"doc_type": "조정사례", "source_id": "주택임대차분쟁조정사례집(2022)", "detail": "수선의무 사례 — 추출 시 사례번호 기입"}], "source": "seed: 임대인 수선의무 전가 유형"}
{"id": "g-004", "text": "임대차 기간은 2년으로 하며, 보증금은 계약 종료와 동시에 반환한다.", "category": "none", "finding_types": ["특이사항 없음"], "evidence": [], "source": "seed: 표준계약서 통상 조항"}
{"id": "g-005", "text": "임차인은 계약갱신요구권을 행사하지 않기로 한다.", "category": "renewal_limit", "finding_types": ["법령과 다름"], "evidence": [{"doc_type": "법령", "source_id": "주택임대차보호법", "detail": "제6조의3, 제10조"}], "source": "seed: 갱신요구권 사전 포기 유형"}
```

- [ ] **Step 5: 실행 → PASS, Commit**

Run: `uv run pytest tests/detect/test_golden.py -v` — Expected: 4 passed

```bash
git add -A && git commit -m "feat: 골든셋 스키마·검증기 및 시드 5건"
```

- [ ] **Step 6: docs/labeling-guide.md 작성 + 30건 확장**

라벨링 가이드에 명시: (1) 1차 소스는 주택임대차분쟁조정사례집 PDF([국토부 정책자료](https://www.molit.go.kr/USR/policyData/m_34681/dtl.jsp?id=4599))와 HUG 전세사기예방센터 유형, (2) 절차 = 사례에서 특약 문구 추출 → 카테고리 부여 → 국가법령정보에서 근거 조문/판례 ID 확정 → evidence 기입, (3) 애매하면 finding_types를 억지로 채우지 말고 notes에 사유 기록 후 보류, (4) AI Hub 데이터 사용 금지. 이 가이드에 따라 pilot.jsonl을 30건 이상으로 확장하고 `scripts/validate_golden.py`로 검증 후 커밋 (`git commit -m "feat: 파일럿 골든셋 30건"`). **이 단계는 사람의 라벨링 작업이 본체다 — 서두르지 말 것.**

---

### Task 6 (P1): PDF 텍스트 추출

**Files:**
- Create: `src/detect/pdf_extract.py`, `tests/detect/test_pdf_extract.py`

**Interfaces:**
- Produces: `extract_text(path: str | Path) -> str`. Task 9의 CLI가 소비.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/detect/test_pdf_extract.py`:
```python
import fitz  # PyMuPDF
import pytest

from src.detect.pdf_extract import extract_text


def _make_pdf(path, texts):
  doc = fitz.open()
  for t in texts:
    page = doc.new_page()
    page.insert_text((72, 72), t)
  doc.save(path)


def test_extracts_multipage_text(tmp_path):
  pdf = tmp_path / "c.pdf"
  _make_pdf(pdf, ["제1조 목적", "특약사항 1. 보증금"])
  out = extract_text(pdf)
  assert "제1조" in out and "특약사항" in out


def test_empty_pdf_raises(tmp_path):
  pdf = tmp_path / "empty.pdf"
  _make_pdf(pdf, [" "])
  with pytest.raises(ValueError):
    extract_text(pdf)
```

- [ ] **Step 2: 실행 → FAIL 확인** — Run: `uv run pytest tests/detect/test_pdf_extract.py -v`

- [ ] **Step 3: 구현**

`src/detect/pdf_extract.py`:
```python
"""계약서 PDF 텍스트 추출. 스캔본(텍스트 레이어 없음)은 P4 OCR 전까지 미지원."""
from pathlib import Path

import fitz


def extract_text(path: str | Path) -> str:
  with fitz.open(path) as doc:
    text = "\n".join(page.get_text() for page in doc)
  if not text.strip():
    raise ValueError("텍스트를 추출할 수 없습니다. 스캔본 PDF는 아직 지원하지 않습니다.")
  return text
```

- [ ] **Step 4: 실행 → PASS, Commit**

```bash
uv run pytest tests/detect/test_pdf_extract.py -v
git add -A && git commit -m "feat: PDF 텍스트 추출"
```

---

### Task 7 (P1): 조항 분해 (segmentation)

**Files:**
- Create: `src/detect/segmenter.py`, `tests/detect/test_segmenter.py`

**Interfaces:**
- Produces: `Clause` dataclass (`index: int, heading: str, text: str`), `segment_clauses(text: str) -> list[Clause]`. Task 8·9·10이 소비.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/detect/test_segmenter.py`:
```python
from src.detect.segmenter import segment_clauses

CONTRACT = """부동산 임대차 계약서
제1조(목적) 임대인은 목적물을 임차인에게 임대한다.
제2조(보증금) 보증금은 1억원으로 한다.
[특약사항]
1. 임차인은 퇴거 시 도배를 새것으로 교체한다.
2. 수리비는 전액 임차인이 부담한다.
"""


def test_segments_articles_and_special_terms():
  clauses = segment_clauses(CONTRACT)
  texts = [c.text for c in clauses]
  assert any("제1조" in c.heading for c in clauses)
  assert any("도배" in t for t in texts)
  assert any("수리비" in t for t in texts)
  # 특약 1번과 2번은 별개 조항으로 분리되어야 한다
  assert not any("도배" in t and "수리비" in t for t in texts)


def test_indices_are_sequential():
  clauses = segment_clauses(CONTRACT)
  assert [c.index for c in clauses] == list(range(len(clauses)))
```

- [ ] **Step 2: 실행 → FAIL 확인** — Run: `uv run pytest tests/detect/test_segmenter.py -v`

- [ ] **Step 3: 구현**

`src/detect/segmenter.py`:
```python
"""규칙 기반 조항 분해. 제N조 및 특약 번호(1. / ① / -)를 경계로 분리."""
import re
from dataclasses import dataclass

_BOUNDARY = re.compile(
  r"(?m)^(?=\s*(?:제\s*\d+\s*조|\[?특약사항\]?|\d+\.\s|[①-⑳]\s?|-\s))"
)
_HEADING = re.compile(r"^(제\s*\d+\s*조(?:\([^)]*\))?|\[?특약사항\]?|\d+\.|[①-⑳]|-)")


@dataclass
class Clause:
  index: int
  heading: str
  text: str


def segment_clauses(text: str) -> list[Clause]:
  chunks = [c.strip() for c in _BOUNDARY.split(text) if c.strip()]
  clauses = []
  for chunk in chunks:
    m = _HEADING.match(chunk)
    heading = m.group(1) if m else ""
    body = chunk[m.end():].strip() if m else chunk
    if heading in ("특약사항", "[특약사항]") and not body:
      continue  # 섹션 제목 단독 줄은 조항이 아님
    if not body:
      continue
    if not heading and clauses == []:
      continue  # 계약서 제목 등 머리말은 조항이 아님
    clauses.append(Clause(index=len(clauses), heading=heading, text=body))
  return clauses
```

- [ ] **Step 4: 실행 → PASS, Commit**

```bash
uv run pytest tests/detect/test_segmenter.py -v
git add -A && git commit -m "feat: 규칙 기반 조항 분해"
```

---

### Task 8 (P1): 조항 카테고리 분류기 (LLM)

**Files:**
- Create: `src/detect/classifier.py`, `tests/detect/test_classifier.py`

**Interfaces:**
- Consumes: Task 4 `Category`, v1 `src.llm.client.llm`.
- Produces: `classify_clause(text: str, llm) -> Category`. Task 9가 소비.

- [ ] **Step 1: 실패하는 테스트 작성 (FakeListChatModel로 LLM 없이)**

`tests/detect/test_classifier.py`:
```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.detect.classifier import classify_clause
from src.detect.taxonomy import Category


def test_parses_valid_category():
  fake = FakeListChatModel(responses=['{"category": "repair_shift"}'])
  assert classify_clause("수리비는 임차인 부담", llm=fake) == Category.REPAIR_SHIFT


def test_invalid_response_falls_back_to_none():
  fake = FakeListChatModel(responses=["잘 모르겠어요"])
  assert classify_clause("아무 조항", llm=fake) == Category.NONE
```

- [ ] **Step 2: 실행 → FAIL 확인** — Run: `uv run pytest tests/detect/test_classifier.py -v`

- [ ] **Step 3: 구현**

`src/detect/classifier.py`:
```python
"""조항 → 위험 카테고리 분류. 판정 아님 — 카테고리는 리트리벌 라우팅 용도."""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.detect.taxonomy import CATEGORY_LABELS, Category

_SYSTEM = (
  "당신은 임대차 계약 조항을 분류한다. 아래 카테고리 id 중 하나만 고른다. "
  "위험 여부를 판단하지 말고 조항의 주제만 분류한다. "
  'JSON으로만 답한다: {"category": "<id>"}\n카테고리: '
  + ", ".join(f"{c.value}({label})" for c, label in CATEGORY_LABELS.items())
)


def classify_clause(text: str, llm) -> Category:
  reply = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=text)])
  content = reply.content if hasattr(reply, "content") else str(reply)
  m = re.search(r"\{.*\}", content, re.S)
  if not m:
    return Category.NONE
  try:
    value = json.loads(m.group())["category"]
    return Category(value)
  except (json.JSONDecodeError, KeyError, ValueError):
    return Category.NONE
```

- [ ] **Step 4: 실행 → PASS, Commit**

```bash
uv run pytest tests/detect/test_classifier.py -v
git add -A && git commit -m "feat: LLM 조항 카테고리 분류기"
```

---

### Task 9 (P1): 발견 유형 판정 + 탐지 파이프라인 + CLI

**Files:**
- Create: `src/detect/findings.py`, `src/detect/pipeline.py`, `scripts/detect.py`, `tests/detect/test_findings.py`

**Interfaces:**
- Consumes: Task 6 `extract_text`, Task 7 `segment_clauses/Clause`, Task 8 `classify_clause`, v1 `QdrantStore.search`(결과 dict에 `score`·`payload`(doc_type/title/source_id/detail 포함)), v1 `resolve_citations`의 URL 생성 로직.
- Produces: `Finding` pydantic 모델(`finding_type: FindingType, doc_type: str, title: str, source_id: str, detail: str, score: float`), `decide_findings(retrieved_docs, threshold) -> list[Finding]`, `run_detection(pdf_path) -> list[dict]` (조항별 `{clause, heading, category, findings}`). Task 10이 소비.

- [ ] **Step 1: 실패하는 테스트 작성 (판정 로직은 순수 함수 — LLM 불필요)**

`tests/detect/test_findings.py`:
```python
from src.detect.findings import decide_findings
from src.detect.taxonomy import FindingType

DOCS = [
  {"score": 0.62, "payload": {"doc_type": "법령", "title": "주택임대차보호법", "source_id": "001", "detail": "제7조"}},
  {"score": 0.55, "payload": {"doc_type": "판례", "title": "대법원 2019다XXXX", "source_id": "002", "detail": ""}},
  {"score": 0.20, "payload": {"doc_type": "법령", "title": "민법", "source_id": "003", "detail": "제623조"}},
]


def test_high_score_docs_become_findings():
  fs = decide_findings(DOCS, threshold=0.5)
  types = {f.finding_type for f in fs}
  assert FindingType.LAW_CONFLICT in types
  assert FindingType.DISPUTE_PRECEDENT in types
  assert all(f.score >= 0.5 for f in fs)


def test_no_hits_means_none_finding():
  fs = decide_findings(DOCS, threshold=0.9)
  assert [f.finding_type for f in fs] == [FindingType.NONE]
```

- [ ] **Step 2: 실행 → FAIL 확인** — Run: `uv run pytest tests/detect/test_findings.py -v`

- [ ] **Step 3: findings 구현**

`src/detect/findings.py`:
```python
"""근거 존재 여부 → 발견 유형 매핑. '위험' 판정 없음 — 근거가 있으면 보여줄 뿐."""
from pydantic import BaseModel

from src.detect.taxonomy import FindingType

_DOC_TYPE_TO_FINDING = {
  "법령": FindingType.LAW_CONFLICT,
  "판례": FindingType.DISPUTE_PRECEDENT,
  "법령해석례": FindingType.DISPUTE_PRECEDENT,
}
# ponytail: 표준계약서와 상이(STANDARD_DIFF)는 P2에서 표준계약서 코퍼스 추가 후 구현


class Finding(BaseModel):
  finding_type: FindingType
  doc_type: str
  title: str
  source_id: str
  detail: str = ""
  score: float


def decide_findings(retrieved_docs: list[dict], threshold: float) -> list[Finding]:
  findings = []
  for doc in retrieved_docs:
    if doc["score"] < threshold:
      continue
    p = doc["payload"]
    ftype = _DOC_TYPE_TO_FINDING.get(p.get("doc_type"))
    if ftype:
      findings.append(Finding(
        finding_type=ftype, doc_type=p["doc_type"], title=p.get("title", ""),
        source_id=p.get("source_id", ""), detail=p.get("detail", ""),
        score=doc["score"],
      ))
  if not findings:
    return [Finding(finding_type=FindingType.NONE, doc_type="", title="",
                    source_id="", detail="", score=0.0)]
  return findings
```

주의: `payload` 키 이름(doc_type/title/source_id/detail)은 v1 인덱서가 실제로 넣는 키와 대조해 맞출 것 — `src/ingest/indexer.py`와 `src/graph/nodes/resolve_citations.py`를 읽고 다르면 **이 코드가 아니라 테스트 픽스처를 실제 키로 수정**.

- [ ] **Step 4: 실행 → PASS 후 파이프라인·CLI 작성**

`src/detect/pipeline.py`:
```python
"""탐지 파이프라인: PDF → 조항 → 분류 → 리트리벌 → 발견 유형."""
from src.detect.classifier import classify_clause
from src.detect.findings import decide_findings
from src.detect.pdf_extract import extract_text
from src.detect.segmenter import segment_clauses
from src.detect.taxonomy import CATEGORY_LABELS

DETECT_THRESHOLD = 0.45  # 파일럿 평가(Task 10)에서 recall 우선으로 재조정


def run_detection(pdf_path: str) -> list[dict]:
  from src.graph.nodes.retrieve import _get_store  # v1 싱글턴 재사용
  from src.llm.client import llm

  store = _get_store()
  results = []
  for clause in segment_clauses(extract_text(pdf_path)):
    category = classify_clause(clause.text, llm=llm)
    docs = store.search(query=clause.text, top_k=5, filters=None)
    findings = decide_findings(docs, threshold=DETECT_THRESHOLD)
    results.append({
      "clause": clause.text,
      "heading": clause.heading,
      "category": CATEGORY_LABELS[category],
      "findings": [f.model_dump() for f in findings],
    })
  return results
```

`scripts/detect.py`:
```python
"""CLI: uv run python scripts/detect.py <계약서.pdf>"""
import json
import sys

from src.detect.pipeline import run_detection

print(json.dumps(run_detection(sys.argv[1]), ensure_ascii=False, indent=2))
print("\n※ 본 결과는 법률 자문이 아닌 정보 제공입니다.", file=sys.stderr)
```

- [ ] **Step 5: 수동 스모크 (integration): 샘플 계약서 PDF 하나로 실행**

Run: `uv run python scripts/detect.py <샘플.pdf>` — Expected: 조항별 JSON + 고지문. (샘플이 없으면 Task 6 테스트의 `_make_pdf` 방식으로 생성)

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: 탐지 파이프라인 및 CLI (근거제시형)"
```

---

### Task 10 (P1): 파일럿 평가 러너

**Files:**
- Create: `src/detect/eval_metrics.py`, `scripts/run_pilot_eval.py`, `tests/detect/test_eval_metrics.py`

**Interfaces:**
- Consumes: Task 5 `load_golden`, Task 9 `decide_findings`·`DETECT_THRESHOLD`, v1 store.
- Produces: `compute_prf(gold: list[set], pred: list[set]) -> dict` (`{"precision", "recall", "f1"}` micro 평균), `reports/pilot-eval.json`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/detect/test_eval_metrics.py`:
```python
from src.detect.eval_metrics import compute_prf


def test_perfect_prediction():
  g = [{"법령과 다름"}, {"특이사항 없음"}]
  assert compute_prf(g, g) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_miss_lowers_recall():
  gold = [{"법령과 다름", "분쟁 전례 있음"}]
  pred = [{"법령과 다름"}]
  m = compute_prf(gold, pred)
  assert m["precision"] == 1.0 and m["recall"] == 0.5
```

- [ ] **Step 2: 실행 → FAIL 확인** — Run: `uv run pytest tests/detect/test_eval_metrics.py -v`

- [ ] **Step 3: 구현**

`src/detect/eval_metrics.py`:
```python
"""발견 유형 micro P/R/F1. 이 도메인은 놓침(recall)이 오탐보다 치명적 (스펙 §8)."""


def compute_prf(gold: list[set], pred: list[set]) -> dict:
  tp = sum(len(g & p) for g, p in zip(gold, pred, strict=True))
  fp = sum(len(p - g) for g, p in zip(gold, pred, strict=True))
  fn = sum(len(g - p) for g, p in zip(gold, pred, strict=True))
  precision = tp / (tp + fp) if tp + fp else 0.0
  recall = tp / (tp + fn) if tp + fn else 0.0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
  return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
```

- [ ] **Step 4: 평가 러너 작성**

`scripts/run_pilot_eval.py`:
```python
"""골든셋으로 탐지 성능 측정: uv run python scripts/run_pilot_eval.py"""
import json
from datetime import date
from pathlib import Path

from src.detect.eval_metrics import compute_prf
from src.detect.findings import decide_findings
from src.detect.golden import load_golden
from src.detect.pipeline import DETECT_THRESHOLD
from src.graph.nodes.retrieve import _get_store

golden = load_golden("data/golden/pilot.jsonl")
store = _get_store()

gold_sets, pred_sets, hits = [], [], 0
for c in golden:
  docs = store.search(query=c.text, top_k=5, filters=None)
  findings = decide_findings(docs, threshold=DETECT_THRESHOLD)
  gold_sets.append({f.value for f in c.finding_types})
  pred_sets.append({f.finding_type.value for f in findings})
  gold_ids = {e.source_id for e in c.evidence}
  got_ids = {d["payload"].get("source_id") for d in docs}
  if not gold_ids or gold_ids & got_ids:
    hits += 1

report = {
  "date": str(date.today()),
  "n": len(golden),
  "threshold": DETECT_THRESHOLD,
  "finding_type_prf": compute_prf(gold_sets, pred_sets),
  "retrieval_hit_at_5": round(hits / len(golden), 4),
}
Path("reports").mkdir(exist_ok=True)
Path("reports/pilot-eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
```

- [ ] **Step 5: 유닛 테스트 PASS 확인 → 골든셋 30건 완성 후 러너 실행**

Run: `uv run pytest tests/detect/test_eval_metrics.py -v` → PASS
Run: `uv run python scripts/run_pilot_eval.py` → `reports/pilot-eval.json` 생성. **첫 수치가 낮아도 정상** — 이 수치가 P2 개선의 베이스라인이다. README에 수치 기록.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: 파일럿 평가 러너 및 베이스라인 기록"
```

---

## P1 완료 기준 (Definition of Done)

- [ ] `uv run pytest -m "not integration"` 전부 PASS, CI 그린
- [ ] 골든셋 30건 이상, `validate_golden.py` 통과, 전 라벨 근거 ID 보유
- [ ] `scripts/detect.py`로 샘플 PDF에서 조항별 발견 유형 JSON 출력
- [ ] `reports/pilot-eval.json` 베이스라인 수치가 README에 기록됨
