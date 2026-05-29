from falkordb import FalkorDB
db = FalkorDB(host="127.0.0.1", port=6379)
g = db.select_graph("onet")

code = "15-2051.00"
prefix = "15-2051"

cypher = """
    MATCH (s:Occupation)
    WHERE s.code STARTS WITH $prefix AND s.code <> $code
    MATCH (s)-[r]->(c)
    WHERE type(r) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
    WITH c, avg(r.score) AS src_score

    WITH collect({c: c, score: src_score}) AS src_vec,
         sum(src_score * src_score) AS src_norm_sq

    UNWIND src_vec AS item
    MATCH (other:Occupation)-[r2]->(c_node)
    WHERE c_node = item.c AND other.code <> $code
      AND type(r2) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
    WITH other, src_norm_sq, sum(item.score * r2.score) AS dot_product

    MATCH (other)-[r3]->(c2)
    WHERE type(r3) IN ['HAS_SKILL', 'HAS_ABILITY', 'HAS_KNOWLEDGE', 'HAS_WORK_ACTIVITY']
    WITH other, dot_product, src_norm_sq, sum(r3.score * r3.score) AS other_norm_sq

    WITH other, (dot_product / (sqrt(src_norm_sq) * sqrt(other_norm_sq))) AS similarity
    RETURN other.code, other.title, similarity
    ORDER BY similarity DESC
    LIMIT 10
"""

try:
    res = g.query(cypher, {"prefix": prefix, "code": code})
    print("Success!")
    for row in res.result_set:
        print(row)
except Exception as e:
    print("Failed with exception:")
    print(e)
