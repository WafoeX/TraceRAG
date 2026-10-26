-- Appended to trusted pg_restore SQL and applied in the SAME transaction.
INSERT INTO application_recovery_state(id,quarantined,reviewed_at)
VALUES (1,true,NULL) ON CONFLICT (id) DO UPDATE
SET quarantined=true, reviewed_at=NULL;
DELETE FROM auth_sessions;
DELETE FROM auth_login_windows;
UPDATE application_budget_accounts SET blocked=true;
-- Never alter spent/held amounts or settle/refund reservations on restore.
UPDATE ingestion_jobs SET state='failed', error_code='restore_review_required',
    lease_token=NULL, lease_until=NULL
WHERE state IN ('queued','parsing','embedding');
DELETE FROM application_request_traces WHERE expires_at <= clock_timestamp();
