import json
import sys
import os

def generateTarget(year, month):
    path = "reddit/ruby/"
    subreddits = ["announcements", "funny", "AskReddit", "todayilearned", "science", "worldnews",
         "pics", "IAmA", "gaming", "videos", "movies", "aww", "Music", "blog", "gifs",
         "news", "explainlikeimfive", "askscience", "EarthPorn", "books"]
    print("Loading Links")
    links = {}
    with open(path + "data/totalLinks", "r") as inFile:
        for line in inFile:
            link = json.loads(line)
            links[link["i"]] = link
    print("Generating Target")
    if year == 2005 and month < 12: return
    if year == 2017 and month > 3: return
    if month < 10: month = "0" + str(month)
    filename = "reddit/{}/RC_{}-{}".format(year,year,month)
    with open(path + filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            if comment["body"].startswith("[removed]")\
                or comment["body"].startswith("[deleted]")\
                or comment["subreddit"] not in subreddits:
                    continue
            if comment["id"] not in links.keys():
                print (comment)
            num_child_comments = links[comment["id"]]["c"]
            if num_child_comments > 20:
                with open(path + "data/target{}{}.json".format(comment["subreddit"],"nc20"), "a") as outFile:
                    outFile.write(json.dumps(comment) + "\n")
                if num_child_comments > 30:
                    with open(path + "data/target{}{}.json".format(comment["subreddit"],"nc30"), "a") as outFile:
                        outFile.write(json.dumps(comment) + "\n")
            if comment["score"] > 1000:
                with open(path + "data/target{}{}.json".format(comment["subreddit"],"sc1000"), "a") as outFile:
                    outFile.write(json.dumps(comment) + "\n")
            if comment["controversiality"] > 2000:
                with open(path + "data/target{}{}.json".format(comment["subreddit"],"ct1000"), "a") as outFile:
                    outFile.write(json.dumps(comment) + "\n")

if __name__ == "__main__":
    generateTarget(int(sys.argv[1]), int(sys.argv[2]))
