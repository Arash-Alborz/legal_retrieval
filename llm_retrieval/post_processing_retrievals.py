import re
import json
from pathlib import Path

results_txt = Path("retrievals/gpt4.1.mini_pw_retrievals_nl.txt") # change name of the file
output_json = Path("retrievals/gpt4.1.mini_pw_retrievals_nl.json") # output

pattern_query = re.compile(r"^query id:\s*(\d+)")
pattern_relevant = re.compile(r"^relevant articles:\s*(.*)")

results_dict = {}

with open(results_txt, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

i = 0
while i < len(lines) - 1:
    m_query = pattern_query.match(lines[i])
    m_relevant = pattern_relevant.match(lines[i+1])
    if m_query and m_relevant:
        qid = m_query.group(1)
        relevant_articles = [x.strip() for x in m_relevant.group(1).split(",") if x.strip()]
        results_dict[qid] = relevant_articles
        i += 2
    else:
        i += 1

with open(output_json, "w", encoding="utf-8") as out:
    json.dump(results_dict, out, indent=2, ensure_ascii=False)

print(f"results_nl.json written to: {output_json}")