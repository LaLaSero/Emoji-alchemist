import json
meta = json.load(open("openmoji/data/openmoji.json"))
for e in meta:
    if e["group"] == "smileys-emotion":
        print(e["hexcode"], e["annotation"])