import json
import os

def countLinks(year, month):
    print("Building Links")
    if year == 2005 and month < 12: return
    if year == 2017 and month > 3: return
    if month < 10: month = "0" + str(month)
    filename = "reddit/ruby/reddit/{}/RC_{}-{}".format(year,year,month)
    if os.path.isfile(filename+"links"): return
    print(filename)
    with open(filename, "r") as inFile, open(filename+"links", "w") as outFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            link = {"i": comment["id"], "p": comment["parent_id"]}
            outFile.write(json.dumps(link) + "\n")
    print("Loading Links")
    links = {}
    for j in range(12):
        month = j+1
        if year == 2005 and month < 12: continue
        if year == 2017 and month > 3: continue
        if month < 10: month = "0" + str(month)
        filename = "reddit/ruby/reddit/{}/RC_{}-{}links".format(year,year,month)
        print(filename)
        with open(filename, "r") as inFile:
            for line in inFile:
                link = json.loads(line)
                link["c"] = 0
                links[link["i"]] = link
    print("Counting Links")
    for i, (_, link) in enumerate(links.iteritems(), 1):
        while link["p"].startswith("t1") and link["p"][3:] in links:
            link = links[link["p"][3:]]
            link["c"] += 1
    with open("reddit/ruby/data/totalLinks", 'w') as outFile:
        for _, link in links.iteritems():
            outFile.write(json.dumps(link) + "\n")
if __name__ == "__main__":
    countLinks(2016, 12)
