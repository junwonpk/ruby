# Ruby prep 3.
# Annotates links with num child comments.

import json
import sys

def loadLinks(filename):
    print "Loading links from {}".format(filename)
    links = {}
    with open(filename, "r") as inFile:
        for line in inFile:
            link = json.loads(line)
            link["c"] = 0
            links[link["i"]] = link

    return links

def annotate(links, outFilename):
    for i, (_, link) in enumerate(links.iteritems(), 1):
        while link["p"].startswith("t1") and link["p"][3:] in links:
            link = links[link["p"][3:]]
            link["c"] += 1
        if i % 100000 == 0:
            print "{} of {} links".format(i, len(links))

    print "Outputting links to {}".format(outFilename)
    with open(outFilename, 'w') as outFile:
        for _, link in links.iteritems():
            outFile.write(json.dumps(link) + "\n")

if __name__ == "__main__":
    links = loadLinks("/media/vvayfarer/ExtraDrive1/bigData/links1/" + sys.argv[1])
    print "Loaded {} links".format(len(links))
    annotate(links, "/media/vvayfarer/ExtraDrive1/bigData/links2/" + sys.argv[1])
