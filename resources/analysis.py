import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)

most_rated = lens.groupby('title').size().sort_values(ascending=False)[:25]
print(most_rated)

"""
SELECT title, count(1)
FROM lens
GROUP BY title
ORDER BY 2 DESC
LIMIT 25;
"""


lens.title.value_counts()[:25]


movie_stats = lens.groupby('title').agg({'rating': [np.size, np.mean]})
movie_stats.head()


# sort by rating average
movie_stats.sort_values([('rating', 'mean')], ascending=False).head()



atleast_100 = movie_stats['rating']['size'] >= 100
movie_stats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:15]

movie_stats = lens.groupby('title').agg({'rating': [np.size, np.mean]})
atleast_100 = movie_stats['rating'].size >= 100
movie_stats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:15]
"""
SELECT title, COUNT(1) size, AVG(rating) mean
FROM lens
GROUP BY title
HAVING COUNT(1) >= 100
ORDER BY 3 DESC
LIMIT 15;
"""

most_50 = lens.groupby('movie_id').size().sort_values(ascending=False)[:50]

"""
CREATE TABLE most_50 AS (
    SELECT movie_id, COUNT(1)
    FROM lens
    GROUP BY movie_id
    ORDER BY 2 DESC
    LIMIT 50
);
"""


"""
SELECT *
FROM lens
WHERE EXISTS (SELECT 1 FROM most_50 WHERE lens.movie_id = most_50.movie_id);
"""


users.age.plot.hist(bins=30)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age');


labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
lens['age_group'] = pd.cut(lens.age, range(0, 81, 10), right=False, labels=labels)
lens[['age', 'age_group']].drop_duplicates()[:10]


lens.groupby('age_group').agg({'rating': [np.size, np.mean]})


lens.set_index('movie_id', inplace=True)
by_age = lens.loc[most_50.index].groupby(['title', 'age_group'])
by_age.rating.mean().head(15)

by_age.rating.mean().unstack(1).fillna(0)[10:20]

by_age.rating.mean().unstack(0).fillna(0)


"""
SELECT title, AVG(IF(sex = 'F', rating, NULL)), AVG(IF(sex = 'M', rating, NULL))
FROM lens
GROUP BY title;
"""

lens.reset_index('movie_id', inplace=True)
pivoted = lens.pivot_table(index=['movie_id', 'title'],
                           columns=['sex'],
                           values='rating',
                           fill_value=0)
pivoted.head()

pivoted['diff'] = pivoted.M - pivoted.F
pivoted.head()

pivoted.reset_index('movie_id', inplace=True)
disagreements = pivoted[pivoted.movie_id.isin(most_50.index)]['diff']
disagreements.sort_values().plot(kind='barh', figsize=[9, 15])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by Men)')
plt.ylabel('Title')
plt.xlabel('Average Rating Difference');


