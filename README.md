# top-post

A python app that predicts which Reddit posts will reach 'top'.

I make requests to the Reddit API to gather information on posts such as time created, score, and number of comments.
I choose some posts to follow and record their early performance at certain intervals of time.
After 24 hours I record their final scores.
I have a thread for each of the posts which will store the data in a file when the post is finished.
Finally, I build a neural network using the early performance as input to predict the final scores.

With this prediction, you can tell that a certain post will reach the 'top' page of reddit.

Prerequisites: Python 3, NumPy, Pandas, Tensorflow, and PRAW.

Run gather_data.py to gather input data and store them to data.pkl.
This will take a long time as it makes API requests periodically, but it is not CPU intensive.
Run ml.py to gather output data, build the model, and train/test it.
