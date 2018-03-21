import os
import csv
import sys

if __name__ == "__main__":
    for b_size in [32, 16, 4]:
        for n_batch in [10000, 100000, 1000000]:
            for t_steps in [10, 40, 160]:
                for n_layers in [2, 4, 8]:
                    for lr_rate in [0.3, 0.003, 0.00003]:
                            # subreddits = ["announcements", "funny", "AskReddit", "todayilearned", "science", "worldnews",
                            #          "pics", "IAmA", "gaming", "videos", "movies", "aww", "Music", "blog", "gifs",
                            #          "news", "explainlikeimfive", "askscience", "EarthPorn", "books"]

                        for sub in ["funny", "news", "todayilearned", "pics", "aww", "books", "gaming"]:
                            for fea in ["nc30", "sc1000", "ct2000"]:
                                os.system(
                                    "python textgen.py {} {} {} {} {} {} {}".format(
                                        lr_rate,
                                        n_layers,
                                        b_size,
                                        t_steps,
                                        n_batch,
                                        sub,
                                        fea
                                    )
                                )
                                # generate(targets, config, data, vocab, startwords, charlens, savefile)
