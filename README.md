## Datasets Used (Not Uploaded Due to Size Limit)

The original datasets used in this capstone project are too large to upload directly to GitHub.
To keep the repository lightweight and within GitHub storage constraints, only sample datasets are included.

You can download the full datasets from the following official sources:

1. TMDB 5000 Movies Dataset
🔗 https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

3. IMDb Reviews Dataset (50,000 Reviews)
🔗(https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?)

5. Films Dataset (for Trend Forecasting)
🔗 https://www.kaggle.com/datasets/thedevastator/popular-movies-database?


## Project Demo Video

The demo video file is also too large to upload to GitHub.
Therefore, the complete walkthrough video has been uploaded externally.

▶️ Demo Video Link (Google Drive)
https://drive.google.com/drive/folders/108Wbwiq83negElaWonxsmz_C2dBk2hej?usp=drive_link


## ***Text, Social Media & Web Analytics Capstone Project – Media & Entertainment Industry***

**Platform:** Jupyter Notebook (Python, NLP, TF-IDF, Text Classification, Trend Analysis, Recommendation Systems)

**Objective:** Analyze audience sentiment, content similarity, engagement behavior, and popularity trends using text mining and social media analytics.

**Project Overview**

This Text & Web Analytics project explores how NLP, sentiment analysis, content similarity, and trend forecasting can be applied to solve real-world problems in the Media & Entertainment industry.

Using TMDB metadata, IMDb reviews, and films datasets, we implemented five practical use cases—from sentiment mining to building a content recommendation system.

The workflow integrates NLP preprocessing, TF-IDF vectorization, classification models, regression forecasting, and similarity-based recommendations.

**Use Cases Summary**
| No.   | Use Case Title                         | Goal                                                          | Technique / Model                                    | Key Result                                                                                   |
| ----- | -------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **1** |  Movie Success Prediction            | Predict if a movie will be Hit or Flop based on metadata.     | Logistic Regression (TF-IDF + Numeric Features)      | Accuracy **80%**. vote_count & popularity are strongest predictors.                          |
| **2** |  Audience Sentiment Analysis         | Classify IMDb reviews as positive or negative.                | NLP Cleaning + TF-IDF + Logistic Regression          | Accuracy **89%**, AUC 0.95. Positive words: “great”, “amazing”. Negative: “worst”, “boring”. |
| **3** |  Trend Forecasting                   | Study and forecast movie popularity trends over time.         | Polynomial Regression + Trend Visualization          | R² 0.67. Popularity stable with genre dominance: Drama, Comedy, Action.                      |
| **4** |  Viewer Engagement Prediction        | Predict high vs low engagement based on votes & popularity.   | Random Forest Classifier                             | Accuracy **98%**. Popularity dominates engagement prediction.                                |
| **5** |  Content Recommendation Optimization | Recommend similar movies using metadata and textual features. | TF-IDF + Cosine Similarity (Content-Based Filtering) | Top 10 recommendations accurate (e.g., Avatar → The Matrix, Apollo 18).                      |

**Overall Insights**

NLP-driven similarity is highly effective for content recommendation.

Sentiment polarity strongly influences audience ratings.

Popularity and vote_count remain the biggest indicators of audience engagement.

Genre trends are consistent globally (Drama, Comedy, Action dominate).

Content-based filtering (TF-IDF + Metadata Soup) provides Netflix-style recommendations.

Text analytics helps understand audience perception, demand, and content patterns.

**Tech Stack**
| Category                       | Tools / Libraries                          |
| ------------------------------ | ------------------------------------------ |
| **NLP Processing**             | NLTK, TF-IDF, WordCloud, Regex, Tokenizers |
| **Machine Learning**           | Logistic Regression, Random Forest         |
| **Vectorization & Similarity** | TF-IDF, CountVectorizer, Cosine Similarity |
| **Visualization**              | Matplotlib, Seaborn                        |
| **Data Sources**               | TMDB, IMDb Reviews, Films Dataset          |
| **Environment**                | Jupyter Notebook (Python)                  |

**Key Visuals in my Use Cases**

Top 20 Frequent Words in Movie Overviews (NLP)

Positive vs Negative Sentiment Keyword Charts

Popularity Trend Forecast

Feature Importance for Engagement Prediction

Top 10 Recommendations for a Selected Movie

Genre Distribution Bar Chart

**Final Outcome**

This project demonstrates how NLP and Text Analytics can support:

-Audience profiling

-Trend forecasting

-Sentiment mining

-Personalized content recommendations

-Engagement prediction

These capabilities mirror real-world systems used by Netflix, IMDb, Prime Video, and Hotstar.

Each use case provides actionable insights supported by text processing, machine learning, and similarity-based ranking.

**Author & Credits**

Project Author: Yashika Saini

Course: Text, Social Media & Web Analytics – Capstone Project

Institution: University of hyderabad

Dataset Sources: TMDB, IMDb, Kaggle
