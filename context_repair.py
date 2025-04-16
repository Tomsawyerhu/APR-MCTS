from framework import get_bug_details
from neo4j_client import Neo4jClient, METHOD_QUERY, LOCAL_VARIABLE_QUERY, IMPORT_QUERY, METHOD_LIST_QUERY
from utils import run_bash

client = Neo4jClient()
script = METHOD_QUERY("org.jfree.chart.plot.MultiplePiePlot.MultiplePiePlot","MultiplePiePlot")
results = client.execute_query(script)
proj = "Chart"
bid = 5
bug = get_bug_details(proj, bid)
path = run_bash("get_source_code_file_path", proj, bid).stdout
line = run_bash("get_first_change_line_count_number", proj, bid).stdout
last_line = run_bash("get_last_change_line_count_number", proj, bid).stdout
for result in results:
    print(result['n']["content"])
    print(result['n']["start_line"],result['n']["end_line"])
    print(result)

query = LOCAL_VARIABLE_QUERY("<init>(org.jfree.data.category.CategoryDataset): org.jfree.chart.plot.MultiplePiePlot")
print(query)
results2 = client.execute_query(query)
print(results2)

import_query=IMPORT_QUERY("org.jfree.chart.plot.MultiplePiePlot")
result3=client.execute_query(import_query)
for result in result3:
    print(result['n']['full_qualified_name'])

method_list_query=METHOD_LIST_QUERY("org.jfree.chart.plot.MultiplePiePlot")
result4=client.execute_query(method_list_query)
for result in result4:
    print(result)
print(bug.code)
print(line)