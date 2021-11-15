SELECT 
comment_id + 3172,
comment_key,
ifnull(author, 0) AS author,
subreddit_key,
link_key,
parent_key,
ifnull(body, 0) AS body,
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
is_valid,
calc_width
FROM comments_iama
WHERE is_valid = 1
INTO OUTFILE '/var/lib/mysql-files/iama_comments.csv'
CHARACTER SET UTF8
FIELDS TERMINATED BY '\t;\t'
LINES TERMINATED BY '\n;\n';