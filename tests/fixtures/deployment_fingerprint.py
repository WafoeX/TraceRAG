"""Synthetic disposable deployment fixture; executed only by opt-in drill."""

import hashlib
import json

from app.database import engine

queries = {
    "users": "SELECT * FROM app_users ORDER BY id",
    "libraries": "SELECT * FROM knowledge_bases ORDER BY id",
    "documents": "SELECT * FROM documents ORDER BY id",
    "chunks": "SELECT id,document_id,chunk_index,content,embedding::text FROM chunks ORDER BY id",
    "logical": "SELECT * FROM logical_documents ORDER BY id",
    "versions": "SELECT * FROM document_versions ORDER BY document_id",
    "pages": "SELECT * FROM chunk_locations ORDER BY chunk_id",
    "accounts": (
        "SELECT key,user_id,limit_cny,spent_cny,held_cny "
        "FROM application_budget_accounts ORDER BY key"
    ),
    "reservations": "SELECT * FROM application_budget_reservations ORDER BY id",
    "events": "SELECT * FROM application_budget_events ORDER BY id",
    "live_trace": (
        "SELECT * FROM application_request_traces WHERE id='11111111-1111-4111-8111-111111111111'"
    ),
    "live_feedback": (
        "SELECT * FROM application_request_feedback "
        "WHERE trace_id='11111111-1111-4111-8111-111111111111'"
    ),
    "job_payloads": (
        "SELECT id,knowledge_base_id,file_hash,dedup_key,title,filename,payload,"
        "attempts,created_at,document_id FROM ingestion_jobs ORDER BY id"
    ),
}
with engine.connect() as c:
    result = {}
    for key, q in queries.items():
        rows = [tuple(r) for r in c.exec_driver_sql(q)]
        result[key] = hashlib.sha256(repr(rows).encode()).hexdigest()
    result["sequence"] = c.exec_driver_sql("SELECT last_value FROM documents_id_seq").scalar_one()
print(json.dumps(result))
