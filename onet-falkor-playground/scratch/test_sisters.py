from falkordb import FalkorDB
db = FalkorDB(host="127.0.0.1", port=6379)
g = db.select_graph("onet")

# Find occupations under 15-2051
res = g.query("MATCH (o:Occupation) WHERE o.code STARTS WITH '15-2051' RETURN o.code, o.title")
print("Occupations starting with 15-2051:")
for row in res.result_set:
    print(f"  {row[0]}: {row[1]}")

# Find competencies count for each
for row in res.result_set:
    code = row[0]
    count_res = g.query("MATCH (o:Occupation {code: $code})-[r]->() RETURN count(r)", {"code": code})
    print(f"  Competencies count for {code}: {count_res.result_set[0][0]}")
