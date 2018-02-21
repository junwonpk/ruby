# Ruby Prep 9
# Annotates timestamp information about comments.
# weekday is from 1: monday to 7: sunday
# time_of_day is as following:
# 0: 0AM to 4AM (Dawn), 2: 4AM to 8AM (Early Morning), 3: 8AM to 12PM (Late Morning),
# 4: 12PM to 4PM (Afternoon), 5: 4PM to 8PM (Evening), 6: 8PM to 12AM (Night)
import json
import sys
import datetime
#from pymongo import MongoClient

def annotateTime(inFilename, outFilename):
    print "Input: {}".format(inFilename)
    print "Output: {}".format(outFilename)

    badCount = 0
    with open(inFilename, "r") as inFile, \
         open(outFilename, "w") as outFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            created = datetime.datetime.fromtimestamp(
                comment["created_utc"]
            )
            comment["weekday"] = created.date().weekday() # monday is 1, sunday is 7.
            time_of_day = created.time().hour / 4 # see above
            comment["time_of_day"] = int(time_of_day)
            outFile.write(json.dumps(comment) + "\n")

            if i % 10000 == 0:
                print "Processed {} lines".format(i)

    print "{} bad comments".format(badCount)

if __name__ == "__main__":
#    collection = MongoClient().reddit.news
#    annotateParents(sys.argv[1], sys.argv[2], collection)
#    path = "../reddit2nd/redditComments2/"
    path = "/home/junwonpk/cloud/cloud/spinel/ruby/reddit2nd/redditComments2/"
    files = ["Reddit2ndTrain", "Reddit2ndDev", "Reddit2ndTest"]
    for f in files:
        i = path + f
        o = path + f + "Time"
        annotateTime(i, o)
