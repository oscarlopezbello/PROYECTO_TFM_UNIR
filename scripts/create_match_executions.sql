-- =============================================================================
-- Tabla match_executions - Trazabilidad de ejecuciones de match vía LLM
-- TFM Match - Proyecto de Maestría UNIR
-- =============================================================================
-- Ejecutar en la misma base de datos que job_requests (analytics_db o la que uses)
-- Ejemplo: mysql -u user -p analytics_db < scripts/create_match_executions.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS match_executions (
    id_execution     INT AUTO_INCREMENT PRIMARY KEY,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Lo que pidió el usuario (request completo enviado al back)
    request_payload  JSON NOT NULL COMMENT 'Payload completo: skills, experience, weights, top_k, hard_filters, etc.',

    -- Lo que se devolvió al front (response completo)
    response_payload JSON NOT NULL COMMENT 'Response: job_request_id, query, results[], collections_enabled, timings_ms',

    -- Metadatos para consultas rápidas (opcional, redundante con JSON)
    top_k            INT COMMENT 'Número de candidatos solicitados',
    num_candidates   INT COMMENT 'Número de candidatos retornados',
    job_request_id   INT COMMENT 'ID en job_requests (si se usa esa tabla)',
    weights_used     JSON COMMENT 'Pesos aplicados por dimensión',

    INDEX idx_created_at (created_at),
    INDEX idx_job_request_id (job_request_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Trazabilidad de ejecuciones de match orquestadas por LLM';
