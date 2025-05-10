import praw
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id= os.getenv("CLIENT_ID"), # Different for every client! Ask user to enter thier id
    client_secret=os.getenv("CLIENT_SECRET"), # Different for every client! Ask user to enter thier id
    user_agent=os.getenv("USER_AGENT")
)

def fetchPost(url):
    # Input URL
    # url = "https://www.reddit.com/r/ipad/comments/1kj3p1n/ipad_mini_any_good/" # It will remain the only input in the program!

    # Extract post ID from URL
    parts = url.split("/")
    post_id = parts[6]  # Example: '1kj3p1n'

    # Fetch submission by ID
    submission = reddit.submission(id=post_id)
    title = submission.title
    body = submission.selftext
    op = submission.author
    score = submission.score
    subreddit = submission.subreddit
    nsfw = submission.over_18
    created = submission.created_utc


    # === Store all the comments in the post ===
    '''
    Alse stores: Author, Created Time, Parent of Comment, ID, link to go to the comment
    '''
    all_comments = []

    # Load all comments, replacing "more"
    submission.comments.replace_more(limit=None)
    # Recursively print all comments
    def storeComments(comments):
        for comment in comments:
            # print("  " * depth + f"u/{comment.author}: {comment.body}\n")
            comm = vars(comment)
            author = comm["author"]
            content = comm["body"]
            created = comm["created"]
            parent = comm["parent_id"]
            id = comm["id"]
            link = f"https://www.reddit.com/r/ipad/comments/{post_id}/comment/{id}/"

            comment_stats = {
                "author": author,
                "content": content,
                "created": created,
                "parent": parent,
                "id": id,
                "link": link
            }

            all_comments.append(comment_stats)
            storeComments(comment.replies)

    storeComments(submission.comments)

    result = {
        "url": url,
        "post_id": post_id,
        "title": title, 
        "body": body, 
        "op": op, 
        "score": score, 
        "subreddit": subreddit, 
        "comment_nest": all_comments,
        "nsfw": nsfw,
        "created": created,
    }

    return result

if __name__ == "__main__":
    post = fetchPost() # All the post details
    
