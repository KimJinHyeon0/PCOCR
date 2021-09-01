SELECT 
comment_id * 2,
comment_key,
ifnull(author, 0) AS author,
subreddit_key,
link_key,
parent_key,
ifnull(body, 0) AS body,
ifnull(depth, 0) AS depth,
UNIX_TIMESTAMP(created_utc)-UNIX_TIMESTAMP('2018-01-01 00:00:00')-32400 AS created_utc,
ifnull(created, 0) AS created,
ifnull(hide_score, 0) AS hide_score,
ifnull(score, 0) AS score,
ifnull(can_gild, 0) AS can_gild,
ifnull(gilded, 0) AS gilded,
ifnull(collapsed, 0) AS collapsed,
ifnull(stickied, 0) AS stickied,
ifnull(controversiality, 0) AS controversiality,
ifnull(distinguished, 0) AS distinguished,
ifnull(vader_score, 0) AS vader_score,
ifnull(vader, 0) AS vader,
is_valid,
ifnull(difficulty, 0) AS difficulty,
ifnull(similarity_post, 0) AS similarity_post,
ifnull(similarity_parent, 0) AS similarity_parent,
calc_width,
calc_depth,
ifnull(inter_comment_time, 0) AS inter_comment_time,
ifnull(prev_comments, 0) AS prev_comments
FROM comments
WHERE is_valid = 1 and UNIX_TIMESTAMP('2018-03-01 00:00:00') - UNIX_TIMESTAMP(created_utc) > 0
INTO OUTFILE '/var/lib/mysql-files/news_comments.csv'
CHARACTER SET UTF8
FIELDS TERMINATED BY '\t;\t'
LINES TERMINATED BY '\n;\n';