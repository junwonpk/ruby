import json
import sys
import os

def generateParentTarget(year, month):
    path = "reddit/ruby/"
    subreddits = ["announcements", "funny", "AskReddit", "todayilearned", "science", "worldnews",
         "pics", "IAmA", "gaming", "videos", "movies", "aww", "Music", "blog", "gifs",
         "news", "explainlikeimfive", "askscience", "EarthPorn", "books"]
    if year == 2005 and month < 12: return
    if year == 2017 and month > 3: return
    if month < 10: month = "0" + str(month)
    filename = "reddit/{}/RC_{}-{}".format(year,year,month)

    # Load Links
    if not os.path.isfile(path + filename + "Links"):
        print("Building Links for {}".format(filename))
        with open(path+filename, "r") as inFile, open(path+filename+"Links", "w") as outFile:
            for i, line in enumerate(inFile, 1):
                comment = json.loads(line)
                if comment["subreddit"] not in subreddits: continue
                link = {"i": comment["id"], "p": comment["parent_id"], "s": comment["subreddit"]}
                outFile.write(json.dumps(link) + "\n")
    links = {}
    if not os.path.isfile(path + filename + "Counts"):
        print("Loading Links for {}".format(filename + "Links"))
        with open(path+filename + "Links", "r") as inFile:
            for line in inFile:
                link = json.loads(line)
                link["c"] = 0
                links[link["i"]] = link
        for i, (_, link) in enumerate(links.items(), 1):
            while link["p"].startswith("t1") and link["p"][3:] in links:
                link = links[link["p"][3:]]
                link["c"] += 1
        with open(path+filename + "Counts", 'w') as outFile:
            for _, link in links.items():
                outFile.write(json.dumps(link) + "\n")
    else:
        with open(path+filename + "Counts", "r") as inFile:
            link = json.loads(line)
            links[link["i"]] = link
    # Load Comments
    print("Loading Comments")
    comments = dict()
    with open(path + filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            if comment["body"].startswith("[removed]")\
                or comment["body"].startswith("[deleted]")\
                or comment["subreddit"] not in subreddits:
                    continue
            num_child_comments = links[comment["id"]]["c"]
            store = {
                "id" : comment["id"],
                "body" : comment["body"],
                "subreddit" : comment["subreddit"],
                "score" : comment["score"],
                "controversiality" : comment["controversiality"],
                "num_child_comments" : num_child_comments,
            }
            comments[comment["id"]] = store

    # Store comments with parents
    with open(path + filename, "r") as inFile:
        for i, line in enumerate(inFile, 1):
            comment = json.loads(line)
            if comment["body"].startswith("[removed]")\
                or comment["body"].startswith("[deleted]")\
                or comment["subreddit"] not in subreddits:
                    continue
            if comment["parent_id"] not in comments.keys():
                continue

            num_child_comments = links[comment["id"]]["c"]
            store = {
                "id" : comment["id"],
                "body" : comment["body"],
                "subreddit" : comment["subreddit"],
                "score" : comment["score"],
                "controversiality" : comment["controversiality"],
                "num_child_comments" : num_child_comments,
                "parent" : comments[comment["parent_id"]]["body"],
            }

            if num_child_comments > 20:
                with open(path + "data/{}{}parent{}{}.json".format(year, month, comment["subreddit"],"nc20"), "a") as outFile:
                    outFile.write(json.dumps(store) + "\n")
                if num_child_comments > 30:
                    with open(path + "data/{}{}parent{}{}.json".format(year, month, comment["subreddit"],"nc30"), "a") as outFile:
                        outFile.write(json.dumps(store) + "\n")
            if comment["score"] > 1000:
                with open(path + "data/{}{}parent{}{}.json".format(year, month, comment["subreddit"],"sc1000"), "a") as outFile:
                    outFile.write(json.dumps(store) + "\n")
            if comment["controversiality"] > 2000:
                with open(path + "data/{}{}parent{}{}.json".format(year, month, comment["subreddit"],"ct1000"), "a") as outFile:
                    outFile.write(json.dumps(store) + "\n")

if __name__ == "__main__":
    generateParentTarget(int(sys.argv[1]), int(sys.argv[2]))
