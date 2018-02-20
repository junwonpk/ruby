# oracle5.py
# Reddit fifth oracle.

import random
import json

print("You are a Human Oracle for Project Ruby.")
print("Please wait while Reddit comments are being loaded.")

skip = 3000
test_size = 20
filename = "RedditDevHead"
outfilename = "responses.txt"
bucket_count = dict()


skipped = 0
comments0 = []
comments1 = []
with open(filename, "r") as data:
    for line in data:
        if skipped < skip:
            skipped += 1
            continue
        if len(comments0) >= test_size and len(comments1) >= test_size:
            break
        comment = json.loads(line)
        if comment['response_time_hours'] > 0.5:
            continue
        if comment['num_child_comments'] == 0:
            if len(comments0) < test_size:
                comments0.append(comment)
        else:
            if len(comments1) < test_size:
                comments1.append(comment)

print("Loaded {} comments".format(len(comments0)))
print("Please read each pair of comments. One of them has no responses, the other has at least 1 response.")
print("Each comment is PRECEDED by some context (its parent comment), the actual comment is the SECOND one.")
print("Pick the comment you think has AT LEAST 1 response.")
answers = []
baseline = 0
for roundNum, (comment0, comment1) in enumerate(zip(comments0, comments1), 1):
    if comment0['response_time_hours'] > comment1['response_time_hours']:
        baseline += 1
    if random.randint(0, 1) == 0:
        comments = [comment0, comment1]
        correctChoice = 2
    else:
        comments = [comment1, comment0]
        correctChoice = 1

    print("")
    print("--------------Comment 1--------------")
    print(format(comments[0]['parent_comment']))
    print("-------------------------------------")
    print(format(comments[0]['body']))
    print("-------------------------------------")
    print("")
    print("--------------Comment 2--------------")
    print(format(comments[1]['parent_comment']))
    print("-------------------------------------")
    print(format(comments[1]['body']))
    print("-------------------------------------")
    print("")
    user = ''
    while user not in [1, 2]:
        print("Round {} of {}".format(roundNum, len(comments0)))
        print("Please enter 1 or 2.")
        print("Remember, pick the comment you think has AT LEAST 1 response.")
        user = input("Your answer: ")
        try:
            user = int(user)
        except ValueError:
            pass
    if user == correctChoice:
        print("You got it correct!")
    else:
        print("You got it wrong.")
    print('')
    answers.append([user, correctChoice])

correct = 0
total = 0
for answer in answers:
    total += 1
    if answer[0] == answer[1]: correct += 1
print('')
print("Precision: {}".format(correct/float(total)))
print("Random Baseline Precision: {}".format(0.5))
print("Response Time Baseline Precision: {}".format(baseline/float(total)))

with open(outfilename, "w") as outfile:
    outfile.write("Correct: {}\n".format(correct))
    outfile.write("Total: {}\n".format(total))
    outfile.write("Precision: {}\n".format(correct/float(total)))
    outfile.write("Random Baseline Precision: {}\n".format(0.5))
    outfile.write("Response Time Baseline Precision: {}\n".format(baseline/float(total)))
    outfile.write("\n")
    for comment0, comment1, answer in zip(comments0, comments1, answers):
        outfile.write(json.dumps(comment0, indent=4) + '\n')
        outfile.write(json.dumps(comment1, indent=4) + '\n')
        outfile.write("Correct" if answer[0] == answer[1] else "Incorrect")
        outfile.write("\n")
        outfile.write("\n")
