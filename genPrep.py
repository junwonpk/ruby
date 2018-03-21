import json
import sys
import os

def countLinks(year, month):
    subreddits = ["announcements", "funny", "AskReddit", "todayilearned", "science", "worldnews",
         "pics", "IAmA", "gaming", "videos", "movies", "aww", "Music", "blog", "gifs",
         "news", "explainlikeimfive", "askscience", "EarthPorn", "books"]

    print("Building Links")
    if year == 2005 and month < 12: return
    if year == 2017 and month > 3: return
    if month < 10: month = "0" + str(month)
    filename = "reddit/ruby/reddit/{}/RC_{}-{}".format(year,year,month)
    if os.path.isfile(filename+"Links"): return
    print("Building Links for {}".format(filename))
    with open(filename, "r") as inFile, open(filename+"Links", "w") as outFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            if comment["subreddit"] not in subreddits: continue
            link = {"i": comment["id"], "p": comment["parent_id"], "s": comment["subreddit"]}
            outFile.write(json.dumps(link) + "\n")

    links = {}
    # if int(year) == 2005 and month < 12: return
    # if int(year) == 2017 and month > 3: return
    # if int(month) < 10: month = "0" + str(month)
    filename = "reddit/ruby/reddit/{}/RC_{}-{}Links".format(year,year,month)
    print("Counting Links for {}".format(filename))
    with open(filename, "r") as inFile:
        for line in inFile:
            link = json.loads(line)
            link["c"] = 0
            links[link["i"]] = link
    for i, (_, link) in enumerate(links.items(), 1):
        while link["p"].startswith("t1") and link["p"][3:] in links:
            link = links[link["p"][3:]]
            link["c"] += 1
    with open("reddit/ruby/data/totalLinks", 'a') as outFile:
        for _, link in links.items():
            outFile.write(json.dumps(link) + "\n")

if __name__ == "__main__":
    countLinks(int(sys.argv[1]), int(sys.argv[2]))
