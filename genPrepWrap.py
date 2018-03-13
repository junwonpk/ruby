import json
import sys
import os

def cleanLinks():

    links = {}

    print("Loading Links")
    with open("reddit/ruby/data/totalLinks", "r") as inFile:
        for line in inFile:
            link = json.loads(line)
            count = 0
            if link["i"] in links.keys():
                count += 1
                links[link["i"]]["c"] += link["c"]
            else:
                links[link["i"]] = link

    print("Found {} duplicate links".format(count))

    print("Saving Links")
    with open("reddit/ruby/data/cleanLinks", 'w') as outFile:
        for _, link in links.items():
            outFile.write(json.dumps(link) + "\n")

if __name__ == "__main__":
    cleanLinks()
