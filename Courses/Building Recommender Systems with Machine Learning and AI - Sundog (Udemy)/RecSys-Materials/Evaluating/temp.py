from MovieLens import MovieLens
from surprise import SVD
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics
import pandas as pd

ml = MovieLens()

print("Loading movie ratings...")
data = ml.loadMovieLensLatestSmall()
print(data)

print("\nComputing movie popularity ranks so we can measure novelty later...")
rankings = ml.getPopularityRanks()
print(rankings)

print("\nComputing item similarities so we can measure diversity later...")
#fullTrainSet = data.build_full_trainset()


