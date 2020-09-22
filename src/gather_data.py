import praw
import threading
import time
import datetime
import numpy as np
import pandas as pd

reddit = praw.Reddit()

# posts = []
threads = []
done = []
pkl_idx = 0
start_time = time.time()

class post:
	checkpoint_times = [x for x in range(2, 62, 2)]
	# checkpoint_times = [1/15]

	def __init__(self, _id, created_utc):
		self.id = _id
		self.time = created_utc
		self.cur_time_idx = 0
		self.scores = []
		self.comments = []

	def utc_to_time(utc):
		return utc

def save_done(done):
	global pkl_idx
	df = pd.DataFrame(data=[[x for x in p.scores] + [x for x in p.comments] + [p.time] for p in done], 
		columns=['scores ' + str(x) for x in post.checkpoint_times] + ['comments ' + str(x) for x in post.checkpoint_times] + ['time'],
		index=[p.id for p in done])

	df.to_pickle('data-{:02d}.pkl'.format(pkl_idx))
	print('saved data ' + str(pkl_idx))
	pkl_idx += 1
	done = []

def manage_post(p, done):
	while p.cur_time_idx < len(post.checkpoint_times):
		now = time.time()
		seconds_to_checkpoint = (p.time + post.checkpoint_times[p.cur_time_idx] * 60) - now
		# print("seconds ", seconds_to_checkpoint)
		if seconds_to_checkpoint > 0:
			time.sleep(seconds_to_checkpoint)
		lock = threading.Lock()
		with lock:
			score = reddit.submission(p.id).score
			num_comments = reddit.submission(p.id).num_comments
		p.scores.append(score)
		p.comments.append(num_comments)
		# print(p.id, p.scores, p.comments)
		p.cur_time_idx += 1
		# print(p.id, " on checkpoint ", p.cur_time_idx)
	lock = threading.Lock()
	with lock:
		done.append(p)
		if len(done) >= 32:
			save_done(done)


for submission in reddit.subreddit("askreddit").stream.submissions():
	now = time.time()
	# print(submission.title, now, submission.created_utc)
	if now - submission.created_utc > 15:
		continue
	cur_post = post(submission.id, now)
	thread = threading.Thread(target=manage_post, args=(cur_post, done))
	threads.append(thread)
	# posts.append(cur_post)
	thread.start()
	if now - start_time > 3*60*60:
		break
for thread in threads:
	thread.join()


# row0 = ['names'] + ['scores ' + str(x) for x in post.checkpoint_times] + ['comments ' + str(x) for x in post.checkpoint_times]
# output = np.array([row0] + [[p.id] + [x for x in p.scores] + [x for x in p.comments] for p in posts])

if done:
	save_done(done)

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)


# for p in posts:
# 	print('{:7s}'.format(p.id), end='')
# 	for score in p.scores:
# 		print('{:4d}'.format(score), end='')
# 	print()
# 	print(' ' * 7, end='')
# 	for num_comments in p.comments:
# 		print('{:4d}'.format(num_comments), end='')
# 	print()