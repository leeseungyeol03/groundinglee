@echo off
set MOCK_LLM=true
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8027
