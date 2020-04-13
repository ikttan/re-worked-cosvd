# Improved recommender system using matrix co-factorization with pruned tags
To improve recommender system by matrix co-factorization with tag genome and time information.

## Dataset
The dataset used in this work is [MovieLens dataset](https://grouplens.org/datasets/movielens/). There are three sets of datasets obtained from [GroupLens](https://grouplens.org).


`ml-latest-small` MovieLens 100K dataset (Year 2016) (Uploaded to this repository)

`mlsmall`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
MovieLens 100K latest dataset (Year 2018) (Uploaded to this repository)


`ml-latest`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[MovieLens 27M latest dataset (Year 2018)](https://grouplens.org/datasets/movielens/latest/)


`ml-latest-2016`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[MovieLens 24M latest dataset (Year 2016)](https://bit.ly/2ULNV5i)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [Surprise package](http://surpriselib.com/).

```bash
pip install surprise
```

This work is also required [cython](https://cython.org/)

## Usage
To generate the User-Tag Matrix & Item-Tag Matrix as the work proposed by [Luo et al.](https://www.sciencedirect.com/science/article/abs/pii/S0957417418307231)
```python
# 'p_ut' is User-Tag Matrix
# 'f_it' is Item-Tag Matrix
# 'tags' is the tags data with tag internal id
# 'ratings' is a joined table of rating data and tag data
import matrices_generation as mg
p_ut, f_it, tags, ratings = mg.generateTagsOrigin(rate, raw_tags)
```

Tag Genome Approach - To utilize the Tag Genome in User-Tag Matrix & Item-Tag Matrix generation
```python
# 'genome_tag' is the tag data with tag id that used for linking the 'genome_score'
# 'genome_score' is the tag data with genome score
p_ut, f_it, tags, ratings = mg.generateTagsWithGenomeScore(rate, raw_tags, genome_tag, genome_score)
```

To initial the co-SVD
```python
algo = co_SVD(n_epochs=20, lr_all=0.006, n_factors=40
              , p_ut=p_ut, f_it=f_it, tags=tags, ratings=ratings)
```

Model Training & Evaluation
```python
## Model Training
algo.fit(trainset)

## Model Testing
predictions = algo.test(testset)

## Model Evaluation
mae = accuracy.mae(predictions, verbose=False)
rmse = accuracy.rmse(predictions, verbose=False)
```
This work utilized Surprise package to build the own prediction algorithm (co-SVD). So, this work supports most of the features provided by Surprise package. For more information, you may check the Surprise package [documentation](https://surprise.readthedocs.io/en/stable/).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
