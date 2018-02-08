import sys
import json

print("You are a Human Oracle for Project Ruby.")
print("Please wait while Reddit comments are being loaded.")

questions = []
test_size = 50
bucket_limit = test_size / float(5)
# bucket_limit = 10
filename = 'redditnews/RedditTrain'
bucket_count = dict()
with open(filename, "r") as data:
    for line in data:
        comment = json.loads(line)
        if len(questions) < test_size:
            bucket = 0
            if comment['num_child_comments'] == 0:
                bucket = 1
            elif comment['num_child_comments'] >= 1 and comment['num_child_comments'] <= 2:
                bucket = 2
            elif comment['num_child_comments'] >= 3 and comment['num_child_comments'] <= 6:
                bucket = 3
            elif comment['num_child_comments'] >= 7 and comment['num_child_comments'] <= 14:
                bucket = 4
            elif comment['num_child_comments'] >= 15:
                bucket = 5
            if bucket not in bucket_count.keys():
                bucket_count[bucket] = 0
            if bucket_count[bucket] < bucket_limit:
                bucket_count[bucket] += 1
                questions.append(comment)
print(bucket_count)


print("Reddit comments loaded.")

print("Please read each comment and write how many comments you think the comment would have been followed by.")
print("Use the following buckets.")
print("1: 0")
print("2: [1,2]")
print("3: [3,6]")
print("4: [7,14]")
print("5: [15+]")
answers = []
for question in questions:
    print("Comment: {}".format(question['body']))
    print('')
    user = ''
    while user not in [1, 2, 3, 4, 5]:
        print("Please write a bucket number")
        print("Use the following buckets.")
        print("1: 0")
        print("2: [1,2]")
        print("3: [3,6]")
        print("4: [7,14]")
        print("5: [15+]")
        user = input("Your answer: ")
        try:
            user = int(user)
        except ValueError:
            pass
    print('')
    bucket = 0
    if question['num_child_comments'] == 0:
        bucket = 1
    elif question['num_child_comments'] >= 1 and question['num_child_comments'] <= 2:
        bucket = 2
    elif question['num_child_comments'] >= 3 and question['num_child_comments'] <= 6:
        bucket = 3
    elif question['num_child_comments'] >= 7 and question['num_child_comments'] <= 14:
        bucket = 4
    elif question['num_child_comments'] >= 15:
        bucket = 5
    answers.append([user, bucket])

correct = 0
total = 0
for answer in answers:
    total += 1
    if answer[0] == answer[1]: correct += 1

print(answers)
print('')
print("Precision: {}".format(correct/float(total)))
