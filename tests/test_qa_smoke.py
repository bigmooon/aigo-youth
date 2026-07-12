import pytest

@pytest.mark.integration
def test_qa_pipeline_end_to_end():
  """Q&A 그래프가 질의에 대해 fallback 또는 인용 포함 답변을 반환한다."""
  from src.graph.pipeline import run

  state = run("보증금 반환이 늦어지면 어떻게 되나요?")
  assert state.get("is_terminated") or state.get("final_answer") or state.get("fallback_message")
