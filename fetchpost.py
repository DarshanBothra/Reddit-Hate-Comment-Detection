import praw
from dotenv import load_dotenv
import os
import predictHS
load_dotenv()

# Initialize Reddit instance

def authorizeUser( CLIENT_ID = '<YOUR-CLIENT-ID>', CLIENT_SECRET  ='<YOUR-CLIENT-SECRET>', USER_AGENT = 'HateCommentIdentifer/0.1 by u/MrAwesome_YT'):
    global reddit
    reddit = praw.Reddit(
    client_id= "RffIzOVmxRTbolDqA0oYLw", # Different for every client! Ask user to enter thier id
    client_secret="ZHhU98b66ER0QzGtWqXIIMNPuWtl0g", # Different for every client! Ask user to enter thier id
    user_agent=USER_AGENT
    )

def fetchPost(url = "https://www.reddit.com/r/NewToReddit/comments/1h9he47/community_recommendations_megathread/"):

    # Input URL
    # url = "https://www.reddit.com/r/ipad/comments/1kj3p1n/ipad_mini_any_good/" # It will remain the only input in the program!

    # Extract post ID from URL
    try:
        parts = url.split("/")
        post_id = parts[6]  # Example: '1kj3p1n'
    except Exception as e:
        return "Invalid POST URL"

    # Fetch submission by ID
    try: 
        submission = reddit.submission(id=post_id)
        title = submission.title
        body = submission.selftext
        op = submission.author
        score = submission.score
        subreddit = submission.subreddit
        nsfw = submission.over_18
        created = submission.created_utc
        tag = predictHS.predict(body)
        code = 200
    except Exception as e:
        return int(str(e).split(" ")[1])

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
            tag = predictHS.predict(content)
            link = f"https://www.reddit.com/r/ipad/comments/{post_id}/comment/{id}/"

            comment_stats = {
                "author": author,
                "content": content,
                "created": created,
                "parent": parent,
                "id": id,
                "link": link,
                "tag": tag,
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
        "tag": tag,
    }

    return result

if __name__ == "__main__":
    print(authorizeUser())
    print(fetchPost())   
