"""pytest 전역 설정.

v1에서 이식된 일부 모듈(src.llm.client, src.api.api)은 import 시점에
환경변수를 즉시 읽어 클라이언트를 초기화한다 (LLM_API_KEY, OC).
import 스모크 테스트(tests/test_imports.py)는 실제 키 없이도 통과해야 하므로,
먼저 .env 파일을 로드하고 (override=False로 기존값 유지), 그 후
값이 없을 때만 더미 값을 채운다 (실제 .env 값을 덮어쓰지 않음).
"""
import os

from dotenv import load_dotenv

load_dotenv(override=False)

os.environ.setdefault("LLM_API_KEY", "test-dummy-key")
os.environ.setdefault("OC", "test-dummy-oc-key")
