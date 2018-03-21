# oracle5.py
# Reddit fifth oracle.

import random
import json

print("You are a Human Oracle for Project Ruby.")
print("Please wait while Reddit comments are being loaded.")

skip = random.randint(1, 9900)
test_size = 20
filename = "RedditDevHead"
outfilename = "responses.txt"


skipped = 0
comments = []
with open(filename, "r") as data:
    for line in data:
        if skipped < skip:
            skipped += 1
            continue
        if len(comments) >= test_size:
            break
        comments.append(json.loads(line))

print("Loaded {} comments".format(len(comments)))
print("Please read the parent comment, followed by the child comment.")
print("Press 0 if you think the CHILD comment has NO responses, press 1 if you think the comment has AT LEAST 1 response.")
answers = []
for roundNum, comment in enumerate(comments, 1):
    print("")
    print("Round {} of {}".format(roundNum, len(comments)))
    print("--------------Parent Comment--------------")
    print(format(comment['parent_comment']))
    print("--------------Actual Comment--------------")
    print(format(comment['body']))
    print("------------------------------------------")
    print("Response Time in hours: {}".format(comment["response_time_hours"]))
    print("------------------------------------------")

    user = ''
    while user not in [0, 1]:
        print("Please enter 0 or 1.")
        print("Remember, 0 if NO responses, 1 if AT LEAST 1 response.")
        try:
            user = raw_input("Your answer: ")
            user = int(user)
        except ValueError, EOFError:
            pass

    correctChoice = 1 if comment["num_child_comments"] > 0 else 0
    if user == correctChoice:
        print("Correct!")
    else:
        print("Incorrect >:)")
    print('')
    answers.append([user, correctChoice])

baseline = 0
correct = 0
total = 0
for answer in answers:
    total += 1
    if answer[0] == answer[1]:
        correct += 1
    if answer[1] == 1:
        baseline += 1
print('')
print("Accuracy: {}".format(correct/float(total)))
print("Naive Baseline Accuracy: {}".format(baseline/float(total)))

with open(outfilename, "w") as outfile:
    outfile.write("Correct: {}\n".format(correct))
    outfile.write("Total: {}\n".format(total))
    outfile.write("Accuracy: {}\n".format(correct/float(total)))
    outfile.write("Naive Baseline Accuracy: {}\n".format(baseline/float(total)))
    outfile.write("\n")
    for comment, answer in zip(comments, answers):
        outfile.write(json.dumps(comment, indent=4) + '\n')
        outfile.write("Response: {} \n".format(answer[0]))
        outfile.write("Correct" if answer[0] == answer[1] else "Incorrect")
        outfile.write("\n")
        outfile.write("\n")
