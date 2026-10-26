-- Reviewed current application bootstrap. Empty database only. No seed users/credits.

CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE app_users (
	id SERIAL NOT NULL, 
	username VARCHAR(64) NOT NULL, 
	password_hash VARCHAR(256) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (username)
)

;


CREATE TABLE application_recovery_state (
	id INTEGER NOT NULL, 
	quarantined BOOLEAN DEFAULT 'false' NOT NULL, 
	reviewed_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	CHECK (id = 1)
)

;


CREATE TABLE auth_login_windows (
	key VARCHAR(100) NOT NULL, 
	attempts INTEGER NOT NULL, 
	PRIMARY KEY (key)
)

;


CREATE TABLE documents (
	id SERIAL NOT NULL, 
	title VARCHAR(512) NOT NULL, 
	source VARCHAR(1024), 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
)

;


CREATE TABLE application_budget_accounts (
	key VARCHAR(64) NOT NULL, 
	user_id INTEGER, 
	limit_cny NUMERIC(30, 12) NOT NULL, 
	spent_cny NUMERIC(30, 12) DEFAULT '0' NOT NULL, 
	held_cny NUMERIC(30, 12) DEFAULT '0' NOT NULL, 
	blocked BOOLEAN DEFAULT 'false' NOT NULL, 
	PRIMARY KEY (key), 
	CHECK (limit_cny >= 0 AND spent_cny >= 0 AND held_cny >= 0), 
	CHECK (spent_cny + held_cny <= limit_cny), 
	CHECK ((key = 'global' AND user_id IS NULL) OR (user_id IS NOT NULL AND key = 'user:' || user_id::text)), 
	UNIQUE (user_id), 
	FOREIGN KEY(user_id) REFERENCES app_users (id)
)

;


CREATE TABLE application_budget_reservations (
	id VARCHAR(36) NOT NULL, 
	user_id INTEGER NOT NULL, 
	role VARCHAR(32) NOT NULL, 
	model VARCHAR(256) NOT NULL, 
	profile_json TEXT NOT NULL, 
	reserved_cny NUMERIC(30, 12) NOT NULL, 
	estimated_cny NUMERIC(30, 12), 
	status VARCHAR(16) NOT NULL, 
	reason VARCHAR(64), 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	CHECK (reserved_cny >= 0 AND (estimated_cny IS NULL OR estimated_cny >= 0)), 
	CHECK (status IN ('held','unknown','settled','released')), 
	CHECK ((status IN ('held','unknown') AND estimated_cny IS NULL) OR (status IN ('settled','released') AND estimated_cny IS NOT NULL)), 
	FOREIGN KEY(user_id) REFERENCES app_users (id)
)

;


CREATE TABLE auth_sessions (
	token_hash VARCHAR(64) NOT NULL, 
	user_id INTEGER NOT NULL, 
	csrf_token VARCHAR(64) NOT NULL, 
	expires_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (token_hash), 
	FOREIGN KEY(user_id) REFERENCES app_users (id) ON DELETE CASCADE
)

;


CREATE TABLE chunks (
	id SERIAL NOT NULL, 
	document_id INTEGER NOT NULL, 
	chunk_index INTEGER NOT NULL, 
	content TEXT NOT NULL, 
	embedding VECTOR(__EMBEDDING_DIM__) NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE
)

;


CREATE TABLE knowledge_bases (
	id SERIAL NOT NULL, 
	owner_id INTEGER NOT NULL, 
	name VARCHAR(200) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(owner_id) REFERENCES app_users (id)
)

;


CREATE TABLE application_budget_events (
	id VARCHAR(36) NOT NULL, 
	action VARCHAR(32) NOT NULL, 
	scope VARCHAR(64) NOT NULL, 
	reservation_id VARCHAR(36), 
	amount_cny NUMERIC(30, 12), 
	reason VARCHAR(512) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(reservation_id) REFERENCES application_budget_reservations (id)
)

;


CREATE TABLE application_request_traces (
	id VARCHAR(36) NOT NULL, 
	user_id INTEGER NOT NULL, 
	knowledge_base_id INTEGER NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	expires_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	content_consent BOOLEAN NOT NULL, 
	metadata_json TEXT NOT NULL, 
	snapshot_json TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES app_users (id) ON DELETE CASCADE, 
	FOREIGN KEY(knowledge_base_id) REFERENCES knowledge_bases (id) ON DELETE CASCADE
)

;


CREATE TABLE chunk_locations (
	chunk_id INTEGER NOT NULL, 
	page_number INTEGER NOT NULL, 
	PRIMARY KEY (chunk_id), 
	FOREIGN KEY(chunk_id) REFERENCES chunks (id) ON DELETE CASCADE
)

;


CREATE TABLE logical_documents (
	id SERIAL NOT NULL, 
	knowledge_base_id INTEGER, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(knowledge_base_id) REFERENCES knowledge_bases (id)
)

;


CREATE TABLE application_request_feedback (
	trace_id VARCHAR(36) NOT NULL, 
	unhelpful BOOLEAN NOT NULL, 
	unsupported_citation BOOLEAN NOT NULL, 
	outdated BOOLEAN NOT NULL, 
	citation_number INTEGER, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (trace_id), 
	CHECK (unhelpful OR unsupported_citation OR outdated), 
	CHECK (citation_number IS NULL OR citation_number BETWEEN 1 AND 20), 
	FOREIGN KEY(trace_id) REFERENCES application_request_traces (id) ON DELETE CASCADE
)

;


CREATE TABLE document_versions (
	document_id INTEGER NOT NULL, 
	logical_id INTEGER NOT NULL, 
	number INTEGER NOT NULL, 
	active BOOLEAN NOT NULL, 
	filename VARCHAR(512), 
	raw_content BYTEA, 
	file_hash VARCHAR(64), 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (document_id), 
	CONSTRAINT uq_document_version_number UNIQUE (logical_id, number), 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE, 
	FOREIGN KEY(logical_id) REFERENCES logical_documents (id) ON DELETE CASCADE
)

;


CREATE TABLE ingestion_jobs (
	knowledge_base_id INTEGER, 
	id SERIAL NOT NULL, 
	file_hash VARCHAR(64) NOT NULL, 
	dedup_key VARCHAR(160) NOT NULL, 
	target_logical_id INTEGER, 
	base_document_id INTEGER, 
	title VARCHAR(512) NOT NULL, 
	filename VARCHAR(512) NOT NULL, 
	payload BYTEA, 
	state VARCHAR(20) NOT NULL, 
	attempts INTEGER NOT NULL, 
	lease_token VARCHAR(64), 
	lease_until TIMESTAMP WITH TIME ZONE, 
	error_code VARCHAR(64), 
	document_id INTEGER, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(knowledge_base_id) REFERENCES knowledge_bases (id), 
	UNIQUE (dedup_key), 
	FOREIGN KEY(target_logical_id) REFERENCES logical_documents (id) ON DELETE CASCADE, 
	FOREIGN KEY(base_document_id) REFERENCES documents (id) ON DELETE CASCADE, 
	FOREIGN KEY(document_id) REFERENCES documents (id)
)

;

CREATE INDEX ix_application_budget_reservations_user_id ON application_budget_reservations (user_id);

CREATE INDEX ix_auth_sessions_expires_at ON auth_sessions (expires_at);

CREATE INDEX ix_chunks_document_id ON chunks (document_id);

CREATE INDEX ix_knowledge_bases_owner_id ON knowledge_bases (owner_id);

CREATE INDEX ix_application_request_traces_expires_at ON application_request_traces (expires_at);

CREATE INDEX ix_application_request_traces_user_id ON application_request_traces (user_id);

CREATE INDEX ix_logical_documents_knowledge_base_id ON logical_documents (knowledge_base_id);

CREATE INDEX ix_document_versions_logical_id ON document_versions (logical_id);

CREATE UNIQUE INDEX uq_document_version_active ON document_versions (logical_id) WHERE active;

CREATE INDEX ix_ingestion_jobs_document_id ON ingestion_jobs (document_id);

CREATE INDEX ix_ingestion_jobs_knowledge_base_id ON ingestion_jobs (knowledge_base_id);

CREATE INDEX ix_ingestion_jobs_state ON ingestion_jobs (state);

CREATE INDEX ix_ingestion_jobs_target_logical_id ON ingestion_jobs (target_logical_id);

CREATE INDEX chunks_content_fts ON chunks USING gin (to_tsvector('english', content));

INSERT INTO application_recovery_state(id,quarantined) VALUES (1,false);
