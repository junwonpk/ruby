# Ruby Prep 6.
# Annotates all in a given file with parent comment if it exists.
# "TOP" if it a top level comment
# null if doesn't exist

import json
import sys
from pymongo import MongoClient

def annotateParents(inFilename, outFilename, collection):
    print "Input: {}".format(inFilename)
    print "Output: {}".format(outFilename)

    badCount = 0
    with open(inFilename, "r") as inFile, \
         open(outFilename, "w") as outFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            parentId = comment["parent_id"]
            if parentId.startswith("t3"):
                print "Shouldn't happen 1"
                # parentComment = {"body": "TOP"}
                continue
            else:
                parentId = parentId[3:]
                parentComment = collection.find_one({"id": parentId})
                if not parentComment:
                    badCount += 1
                    continue
            comment["parent_comment"] = parentComment["body"]
            comment["response_time_hours"] = (comment["created_utc"] - parentComment["created_utc"]) / 60.0 / 60.0
            outFile.write(json.dumps(comment) + "\n")

            if i % 10000 == 0:
                print "Processed {} lines".format(i)

    print "{} bad comments".format(badCount)

if __name__ == "__main__":
    collection = MongoClient().reddit.news
    annotateParents(sys.argv[1], sys.argv[2], collection)
