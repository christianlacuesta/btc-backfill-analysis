WITH pattern_match_stats AS (
  SELECT
    p.id AS pattern_id,
    COUNTIF(p.id IN (m.pattern_id_1, m.pattern_id_2)) AS reoccurrence,
    AVG(
      CASE
        WHEN p.id = m.pattern_id_1 OR p.id = m.pattern_id_2
        THEN m.sim_score
      END
    ) AS avg_similarity
  FROM `bitcoin-480204.crypto.btc_pattern_1h` p
  LEFT JOIN `bitcoin-480204.crypto.btc_pattern_self_matches` m
    ON p.id = m.pattern_id_1 OR p.id = m.pattern_id_2
  GROUP BY p.id
),
strength_norm AS (
  SELECT
    pattern_id,
    reoccurrence,
    avg_similarity,
    SAFE_DIVIDE(
      LOG(1 + reoccurrence),
      MAX(LOG(1 + reoccurrence)) OVER()
    ) AS reoccurrence_norm
  FROM pattern_match_stats
),
occurrences AS (
  SELECT
    id AS pattern_id,
    signal_ts,
    DATE_DIFF(CURRENT_DATE(), DATE(signal_ts), DAY) AS days_ago
  FROM `bitcoin-480204.crypto.btc_pattern_1h`
),
hotness_raw AS (
  SELECT
    pattern_id,
    SUM(EXP(-days_ago / 7.0)) AS recency_weighted_hits
  FROM occurrences
  GROUP BY pattern_id
),
hotness_norm AS (
  SELECT
    pattern_id,
    recency_weighted_hits,
    SAFE_DIVIDE(
      recency_weighted_hits,
      MAX(recency_weighted_hits) OVER()
    ) AS hotness_score
  FROM hotness_raw
)
SELECT
  s.pattern_id,
  s.reoccurrence,
  s.avg_similarity,
  -- strength score: 50% similarity + 50% normalized reoccurrence
  0.5 * s.avg_similarity + 0.5 * s.reoccurrence_norm AS strength_score,
  h.hotness_score
FROM strength_norm s
LEFT JOIN hotness_norm h
  ON s.pattern_id = h.pattern_id
ORDER BY strength_score DESC limit 20;
