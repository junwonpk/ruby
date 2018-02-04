import json
import sys

def filterNews():
    key = 'subreddit'
    value = 'news'
    months = ['01', '02', '03', '04', '05', '06']
    path = 'reddit/2016/'
    for month in months:
        inFilename = path + 'RC_2016-' + month
        outFilename = path + 'RC-2016-' + month + 'N'
        print "Processing {} to {}".format(inFilename, outFilename)
        with open(inFilename, "r") as inFile, open(outFilename, "w") as outFile:
            for i, line in enumerate(inFile, 1):
                comment = json.loads(line)
                if comment[key] == value:
                    outFile.write(json.dumps(comment) + "\n")
                if i % 10000000 == 0:
                    print "Processed {} lines".format(i)

if __name__ == "__main__":
    filterNews()
