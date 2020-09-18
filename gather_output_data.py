import pandas as pd
import praw

dataset_path = 'big_data_start_9-16-2020-1630-1930'
input_df = pd.read_pickle(dataset_path + '/data-951.pkl')

# print(input_df.index)

reddit = praw.Reddit()
# post = reddit.submission(id='iu4hhw')
# print(dir(post))
# exit()

columns = ['end score', 'end comments']
data = {column : [] for column in columns}
index = []
for post_id in input_df.index:
	index.append(post_id)
	post = reddit.submission(id=post_id)
	data['end score'].append(post.score)
	data['end comments'].append(post.num_comments)
	if len(index) % 100 == 99:
		print(len(index), " done")

output_df = pd.DataFrame(data, columns=columns, index=index)

output_df.to_pickle('output_data.pkl')
print('saved to output_data.pkl')