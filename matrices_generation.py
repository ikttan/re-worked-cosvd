import pandas as pd
import numpy as np
import math

## Generate Matrices with the formula proposed in (Luo et al., 2019)
def generateTagsOrigin(rate, tags):
    temp_df = rate
    temp_df = temp_df.iloc[:,0:3]

    uni_tags = tags.tag.unique()
    uni_tags = {'tag':uni_tags, 'tid':range(len(uni_tags))}
    u_tag = pd.DataFrame(uni_tags)
    tags = pd.merge(tags, u_tag, how='left', on=['tag'])

    w = tags.groupby(['tid', 'movieId'], as_index=False)['userId'].count()
    w.columns = ['tid', 'movieId', 'cn']
    temp = w.groupby('movieId', as_index=False)['cn'].sum()
    temp = temp.set_index('movieId')
    iteration = range(len(w))
    w['val'] = np.array(pd.Series(iteration).map(lambda x: w.cn[x] / temp.loc[w.movieId[1], 'cn']))

    f = tags.groupby(['userId', 'tid'], as_index=False).agg({'movieId': 'count'})
    f.columns = ['userId', 'tid', 'cn']
    temp = f.groupby('userId', as_index=False)['cn'].sum()
    temp = temp.set_index('userId')
    iteration = range(len(f))
    f['val'] = np.array(pd.Series(iteration).map(lambda x: f.cn[x] / temp.loc[f.userId[x], 'cn']))

    nl_alpha = -0.006

    nl_ut = tags.groupby(['userId', 'tid'], as_index=False).agg({'timestamp': 'max'})
    nl_ut = nl_ut.sort_values(by=['userId', 'timestamp'], ascending=[True,False]).reset_index(drop=True)
    g = nl_ut.groupby(['userId'])
    nl_ut['times'] = g['timestamp'].rank(method='first', ascending=False) - 1
    nl_ut['val'] = nl_alpha * nl_ut['times']
    nl_ut['val'] = nl_ut['val'].map(lambda x: math.exp(x)).tolist()

    nl_it = tags.groupby(['movieId', 'tid'], as_index=False).agg({'timestamp': 'max'})
    nl_it = nl_it.sort_values(by=['movieId', 'timestamp'], ascending=[True,False]).reset_index(drop=True)
    g = nl_it.groupby(['movieId'])
    nl_it['times'] = g['timestamp'].rank(method='first', ascending=False) - 1
    nl_it['val'] = nl_alpha * nl_it['times']
    nl_it['val'] = nl_it['val'].map(lambda x: math.exp(x)).tolist()

    ru = temp_df.groupby(['userId'], as_index=False).agg({'rating': 'mean'})
    ru = ru.rename(index=str, columns={"rating": "ru"})
    temp = f[['userId']].drop_duplicates()
    ru = pd.merge(temp, ru, how="left", on=['userId'])
    ru.fillna(0, inplace=True)

    p_ut = pd.DataFrame(f[['userId','tid']], columns=['userId','tid'])
    how='outer'
    overall = pd.merge(temp_df, tags, how=how, on=['userId', 'movieId'])
    overall = overall.drop(columns=['tag','timestamp'])
    overall.rating.fillna(0, inplace=True)
    temp_rt = overall[ (-pd.isna(overall.userId)) & (-pd.isna(overall.tid))]
    rt = overall.groupby(['userId', 'tid'], as_index=False).agg({'rating': 'mean'})
    rt = rt.rename(index=str, columns={"rating": "rt"})
    overall = pd.merge(overall, w, how=how, on=['movieId', 'tid'])
    overall = overall.drop(columns=['cn'])
    overall = overall.rename(index=str, columns={"val": "w_it"})
    overall.w_it.fillna(0, inplace=True)
    overall = pd.merge(overall, ru, how=how, on=['userId'])
    overall.ru.fillna(0, inplace=True)
    overall['r_bias'] = overall.rating - overall.ru
    overall['b_it'] = overall.r_bias * overall.w_it

    b_it = overall[~pd.isna(overall.tid)].groupby(['userId', 'tid'], as_index=False).agg({'w_it': 'sum', 'b_it':'sum'})
    b_it['val'] = b_it.b_it / b_it.w_it

    ru = ru.set_index('userId')
    rt = rt.set_index(['userId', 'tid'])
    f = f.set_index(['userId', 'tid'])
    b_it = b_it.set_index(['userId', 'tid'])
    nl_ut = nl_ut.set_index(['userId', 'tid'])

    p_ut['val'] = list(map(lambda x,y: ru.loc[x, 'ru'] + b_it.loc[(x, y), 'val']
              + 1.7 * f.loc[(x,y),'val'] * (rt.loc[(x,y), 'rt'] - ru.loc[x, 'ru'])
              + 0.05 * nl_ut.loc[(x,y), 'val'] , p_ut.userId, p_ut.tid))

    f_it = pd.DataFrame(w[['movieId','tid']], columns=['movieId','tid'])
    w = w.set_index(['movieId', 'tid'])
    nl_it = nl_it.set_index(['movieId', 'tid'])
    f_it['val'] = list(map(lambda x,y: w.loc[(x,y), 'val'] + 0.05 * nl_it.loc[(x,y), 'val'], f_it.movieId, f_it.tid))

    ratings = overall.iloc[:,0:4]
    return p_ut, f_it, tags, ratings



## Tag Genome Approach
## Apply the Tag genome into the Matrices Generation
## Tag Genome was applied for forming the Item-Tag Matrix
def generateTagsWithGenomeScore(rate, tags, genome_tag, genome_score):
    temp_df = rate
    temp_df = temp_df.iloc[:,0:3]

    tags = pd.merge(tags, genome_tag, how='left', on=['tag'])
    tags = tags[~np.isnan(tags.tagId)]
    tags = tags[tags.movieId.isin(genome_score.movieId) & tags.tagId.isin(genome_score.tagId)]
    
    uni_tags = tags.groupby(['tag'], as_index=False)['userId'].count().tag
    tags = tags[tags.tag.isin(uni_tags)].reset_index(drop=True)
    uni_tags = {'tag':uni_tags, 'tid':range(len(uni_tags))}
    u_tag = pd.DataFrame(uni_tags)
    tags = pd.merge(tags, u_tag, how='left', on=['tag'])
    
    f_it = genome_score[genome_score.movieId.isin(tags.movieId) & genome_score.tagId.isin(tags.tagId)]
    temp_tag = tags.iloc[:, -2:]
    temp_tag = temp_tag.drop_duplicates()
    f_it = pd.merge(f_it, temp_tag, how='left', on=['tagId'])
    f_it = f_it.rename(index=str, columns={"relevance": "val"})
    
    f_it = f_it.drop(['tagId'], axis=1)
    tags = tags.drop(['tagId'], axis=1)
    
    w = tags.groupby(['tid', 'movieId'], as_index=False)['userId'].count()
    w.columns = ['tid', 'movieId', 'cn']
    temp = w.groupby('movieId', as_index=False)['cn'].sum()
    temp = temp.set_index('movieId')
    iteration = range(len(w))
    w['val'] = pd.Series(iteration).map(lambda x: w.cn[x] / temp.loc[w.movieId[1], 'cn']).tolist()
    
    f = tags.groupby(['userId', 'tid'], as_index=False).agg({'movieId': 'count'})
    f.columns = ['userId', 'tid', 'cn']
    temp = f.groupby('userId', as_index=False)['cn'].sum()
    temp = temp.set_index('userId')
    iteration = range(len(f))
    f['val'] = pd.Series(iteration).map(lambda x: f.cn[x] / temp.loc[f.userId[x], 'cn']).tolist()
    
    nl_alpha = -0.006
    
    nl_ut = tags.groupby(['userId', 'tid'], as_index=False).agg({'timestamp': 'max'})
    nl_ut = nl_ut.sort_values(by=['userId', 'timestamp'], ascending=[True,False]).reset_index(drop=True)
    g = nl_ut.groupby(['userId'])
    nl_ut['times'] = g['timestamp'].rank(method='first', ascending=False)
    nl_ut['val'] = nl_alpha * nl_ut['times']
    nl_ut['val'] = nl_ut['val'].map(lambda x: math.exp(x)).tolist()
    
    ru = temp_df.groupby(['userId'], as_index=False).agg({'rating': 'mean'})
    ru = ru.rename(index=str, columns={"rating": "ru"})
    p_ut = pd.DataFrame(f[['userId','tid']], columns=['userId','tid'])
    how='outer'
    overall = pd.merge(temp_df, tags, how=how, on=['userId', 'movieId'])
    overall = overall.drop(columns=['tag','timestamp'])
    overall.rating.fillna(0, inplace=True)
    temp_rt = overall[ (-pd.isna(overall.userId)) & (-pd.isna(overall.tid))]
    rt = overall.groupby(['userId', 'tid'], as_index=False).agg({'rating': 'mean'})
    rt = rt.rename(index=str, columns={"rating": "rt"})
    overall = pd.merge(overall, w, how=how, on=['movieId', 'tid'])
    overall = overall.drop(columns=['cn'])
    overall = overall.rename(index=str, columns={"val": "w_it"})
    overall.w_it.fillna(0, inplace=True)
    overall = pd.merge(overall, ru, how=how, on=['userId'])
    overall.ru.fillna(0, inplace=True)
    overall['r_bias'] = overall.rating - overall.ru
    overall['b_it'] = overall.r_bias * overall.w_it
    b_it = overall[-pd.isna(overall.tid)].groupby(['userId', 'tid'], as_index=False).agg({'w_it': 'sum', 'b_it':'sum'})
    b_it['val'] = b_it.b_it / b_it.w_it
    
    ru = ru.set_index('userId')
    rt = rt.set_index(['userId', 'tid'])
    f = f.set_index(['userId', 'tid'])
    b_it = b_it.set_index(['userId', 'tid'])
    nl_ut = nl_ut.set_index(['userId', 'tid'])

    p_ut['val'] = list(map(lambda x,y: ru.loc[x, 'ru'] + b_it.loc[(x, y), 'val'] 
              + 1.7 * f.loc[(x,y),'val'] * (rt.loc[(x,y), 'rt'] - ru.loc[x, 'ru']) 
              + 0.05 * nl_ut.loc[(x,y), 'val'] , p_ut.userId, p_ut.tid))
    
    ratings = overall.iloc[:,0:4]
    return p_ut, f_it, tags, ratings



## Hybrid Approach
## Generate Item-Tag Matrix with tag genome, tag frequency and time information & User-Tag Matrix
## For those tag without tag genome, the original weightage calculation will be applied 
## else it will subtitute with tag genome 
def generateTagsOriginWithGenomeWeightage(rate, tags, genome_tag, genome_score):
    temp_df = rate
    temp_df = temp_df.iloc[:,0:3]

    tags = pd.merge(tags, genome_tag, how='left', on=['tag'])
    tags = tags[~np.isnan(tags.tagId)]
    tags = tags[tags.movieId.isin(genome_score.movieId) & tags.tagId.isin(genome_score.tagId)]
    
    uni_tags = tags.groupby(['tag'], as_index=False)['userId'].count().tag
    tags = tags[tags.tag.isin(uni_tags)].reset_index(drop=True)
    uni_tags = {'tag':uni_tags, 'tid':range(len(uni_tags))}
    u_tag = pd.DataFrame(uni_tags)
    tags = pd.merge(tags, u_tag, how='left', on=['tag'])
    
    temp_tag = tags.iloc[:, -2:]
    temp_tag = temp_tag.drop_duplicates()
    tags = tags.drop(['tagId'], axis=1)
    
    w = tags.groupby(['tid', 'movieId'], as_index=False)['userId'].count()
    w.columns = ['tid', 'movieId', 'cn']
    temp = w.groupby('movieId', as_index=False)['cn'].sum()
    temp = temp.set_index('movieId')
    iteration = range(len(w))
    w['val'] = pd.Series(iteration).map(lambda x: w.cn[x] / temp.loc[w.movieId[1], 'cn']).tolist()
    
    f = tags.groupby(['userId', 'tid'], as_index=False).agg({'movieId': 'count'})
    f.columns = ['userId', 'tid', 'cn']
    temp = f.groupby('userId', as_index=False)['cn'].sum()
    temp = temp.set_index('userId')
    iteration = range(len(f))
    f['val'] = pd.Series(iteration).map(lambda x: f.cn[x] / temp.loc[f.userId[x], 'cn']).tolist()
    
    nl_alpha = -0.006
    
    nl_ut = tags.groupby(['userId', 'tid'], as_index=False).agg({'timestamp': 'max'})
    nl_ut = nl_ut.sort_values(by=['userId', 'timestamp'], ascending=[True,False]).reset_index(drop=True)
    g = nl_ut.groupby(['userId'])
    nl_ut['times'] = g['timestamp'].rank(method='first', ascending=False)
    nl_ut['val'] = nl_alpha * nl_ut['times']
    nl_ut['val'] = nl_ut['val'].map(lambda x: math.exp(x)).tolist()
    
    nl_it = tags.groupby(['movieId', 'tid'], as_index=False).agg({'timestamp': 'max'})
    nl_it = nl_it.sort_values(by=['movieId', 'timestamp'], ascending=[True,False]).reset_index(drop=True)
    g = nl_it.groupby(['movieId'])
    nl_it['times'] = g['timestamp'].rank(method='first', ascending=False)
    nl_it['val'] = nl_alpha * nl_it['times']
    nl_it['val'] = nl_it['val'].map(lambda x: math.exp(x)).tolist()
    
    ru = temp_df.groupby(['userId'], as_index=False).agg({'rating': 'mean'})
    ru = ru.rename(index=str, columns={"rating": "ru"})
    p_ut = pd.DataFrame(f[['userId','tid']], columns=['userId','tid'])
    how='outer'
    overall = pd.merge(temp_df, tags, how=how, on=['userId', 'movieId'])
    overall = overall.drop(columns=['tag','timestamp'])
    overall.rating.fillna(0, inplace=True)
    temp_rt = overall[ (-pd.isna(overall.userId)) & (-pd.isna(overall.tid))]
    rt = overall.groupby(['userId', 'tid'], as_index=False).agg({'rating': 'mean'})
    rt = rt.rename(index=str, columns={"rating": "rt"})
    overall = pd.merge(overall, w, how=how, on=['movieId', 'tid'])
    overall = overall.drop(columns=['cn'])
    overall = overall.rename(index=str, columns={"val": "w_it"})
    overall.w_it.fillna(0, inplace=True)
    overall = pd.merge(overall, ru, how=how, on=['userId'])
    overall.ru.fillna(0, inplace=True)
    overall['r_bias'] = overall.rating - overall.ru
    overall['b_it'] = overall.r_bias * overall.w_it
    b_it = overall[-pd.isna(overall.tid)].groupby(['userId', 'tid'], as_index=False).agg({'w_it': 'sum', 'b_it':'sum'})
    b_it['val'] = b_it.b_it / b_it.w_it
    
    ru = ru.set_index('userId')
    rt = rt.set_index(['userId', 'tid'])
    f = f.set_index(['userId', 'tid'])
    b_it = b_it.set_index(['userId', 'tid'])
    nl_ut = nl_ut.set_index(['userId', 'tid'])

    p_ut['val'] = list(map(lambda x,y: ru.loc[x, 'ru'] + b_it.loc[(x, y), 'val'] 
              + 1.7 * f.loc[(x,y),'val'] * (rt.loc[(x,y), 'rt'] - ru.loc[x, 'ru']) 
              + 0.05 * nl_ut.loc[(x,y), 'val'] , p_ut.userId, p_ut.tid))
    
    genome_score = genome_score[genome_score.movieId.isin(tags.movieId) & genome_score.tagId.isin(temp_tag.tagId)]
    genome_score = pd.merge(genome_score, temp_tag, how='left', on=['tagId'])
    genome_score = genome_score.rename(index=str, columns={"relevance": "val"})
    
    f_it = pd.DataFrame(w[['movieId','tid']], columns=['movieId','tid'])
    w = w.set_index(['movieId', 'tid'])
    genome_score = genome_score.set_index(['movieId', 'tid'])
    nl_it = nl_it.set_index(['movieId', 'tid'])
    f_it['val'] = list(map(lambda x,y: genome_score.loc[(x,y), 'val'] + 0.05 * nl_it.loc[(x,y), 'val'] if (x,y) in genome_score.index else w.loc[(x,y), 'val'] + 0.05 * nl_it.loc[(x,y), 'val'], f_it.movieId, f_it.tid))
    
    ratings = overall.iloc[:,0:4]
    return p_ut, f_it, tags, ratings
