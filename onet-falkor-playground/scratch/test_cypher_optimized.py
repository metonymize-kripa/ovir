from falkordb import FalkorDB
import time
db = FalkorDB(host="127.0.0.1", port=6379)
g = db.select_graph("onet")

code = "15-2051.00"
prefix = "15-2051"

cypher = """
    MATCH (s:Occupation)
    WHERE s.code STARTS WITH $prefix AND s.code <> $code
    MATCH (s)-[r1]->(c)
    WHERE type(r1) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
    WITH c, avg(r1.score) AS avg_src_score
    WITH sum(avg_src_score * avg_src_score) AS src_norm_sq

    MATCH (s:Occupation)
    WHERE s.code STARTS WITH $prefix AND s.code <> $code
    MATCH (s)-[r1]->(c)<-[r2]-(other:Occupation)
    WHERE type(r1) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
      AND type(r2) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
      AND other.code <> $code
    WITH src_norm_sq, other, c, avg(r1.score) AS avg_src_score, r2.score AS other_score
    WITH src_norm_sq, other, sum(avg_src_score * other_score) AS dot_product

    MATCH (other)-[r3]->(c2)
    WHERE type(r3) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
    WITH other, dot_product, src_norm_sq, sum(r3.score * r3.score) AS other_norm_sq

    WITH other, (dot_product / (sqrt(src_norm_sq) * sqrt(other_norm_sq))) AS similarity
    RETURN other.code, other.title, similarity
    ORDER BY similarity DESC
    LIMIT 10
"""

t0 = time.perf_counter()
try:
    res = g.query(cypher, {"prefix": prefix, "code": code})
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Success! Executed in {elapsed:.2f}ms")
    for row in res.result_set:
        print(row)
except Exception as e:
    print("Failed with exception:")
    print(e)
