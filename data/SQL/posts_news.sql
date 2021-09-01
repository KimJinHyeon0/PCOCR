SELECT 
post_id * 2 - 1,
post_key,
ifnull(author, 0) AS author,
subreddit_key,
ifnull(title, 0) AS title,
ifnull(url, 0) AS url,
ifnull(domain, 0) AS domain,
ifnull(selftext, 0) AS selftext,
UNIX_TIMESTAMP(created_utc)-UNIX_TIMESTAMP('2018-01-01 00:00:00')-32400 AS created_utc,
ifnull(created, 0) AS created,
ifnull(hide_score, 0) AS hide_score,
ifnull(score, 0) AS score,
ifnull(can_gild, 0) AS can_gild,
ifnull(gilded, 0) AS gilded,
ifnull(over_18, 0) AS over_18,
ifnull(stickied, 0) AS stickied,
ifnull(is_video, 0) AS is_video,
ifnull(is_self, 0) AS is_self,
ifnull(locked, 0) AS locked,
ifnull(distinguished, 0) AS distinguished,
ifnull(vader_score, 0) AS vader_score,
ifnull(vader, 0) AS vader,
ifnull(difficulty, 0) AS difficulty,
calc_width,
volume,
is_valid
FROM posts
WHERE is_valid = 1 and UNIX_TIMESTAMP('2018-03-01 00:00:00') - UNIX_TIMESTAMP(created_utc) > 0
INTO OUTFILE '/var/lib/mysql-files/news_posts.csv'
CHARACTER SET UTF8
FIELDS TERMINATED BY '\t;\t'
LINES TERMINATED BY '\n;\n';