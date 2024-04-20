import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
models = ('1. Course Similarity',
          '2. User Profile',
          '3. Clustering',
          '4. Clustering with PCA',
          '5. KNN',
          '6. NMF',
          '7. Neural Network')

RANDOM_SEED = 42
NUM_GENRES = 14


def load_ratings():
    return pd.read_csv('ratings.csv')


def load_course_sims():
    return pd.read_csv('sim.csv')


def load_courses():
    df = pd.read_csv('course_processed.csv')
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv('courses_bows.csv')


def load_course_genres():
    return pd.read_csv('course_genres.csv')


def load_user_profiles():
    return pd.read_csv('user_profiles.csv')


def get_model_index(model_name):
    index = None
    for i in range(len(models)):
        if model_name ==models[i]:
            index = i
            break
    return index
        


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['course'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv('ratings.csv', index=False)
        return new_id


def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(
        ['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),)

        self.user_bias = layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            name='user_bias',)
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),)
        self.item_bias = layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            name='item_bias',)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        x = dot_user_item + user_bias + item_bias
        return tf.nn.relu(x)


def encode_ratings(raw_data):
    encoded_data = raw_data.copy()
    user_list = encoded_data['user'].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}

    course_list = encoded_data['item'].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    encoded_data['user'] = encoded_data['user'].map(user_id2idx_dict)
    encoded_data['item'] = encoded_data['item'].map(course_id2idx_dict)
    encoded_data['rating'] = encoded_data['rating'].values.astype('int')

    return encoded_data, user_idx2id_dict, course_idx2id_dict


def generate_train_test_datasets_ann(dataset, scale=True):

    min_rating = min(dataset["rating"])
    max_rating = max(dataset["rating"])

    dataset = dataset.sample(frac=1, random_state=42)
    x = dataset[["user", "item"]].values
    if scale:
        y = dataset["rating"].apply(lambda x: (
            x - min_rating) / (max_rating - min_rating)).values
    else:
        y = dataset["rating"].values

    train_indices = int(0.8 * dataset.shape[0])
    test_indices = int(0.9 * dataset.shape[0])

    x_train, x_val, x_test, y_train, y_val, y_test = (
        x[:train_indices],
        x[train_indices:test_indices],
        x[test_indices:],
        y[:train_indices],
        y[train_indices:test_indices],
        y[test_indices:],
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def train_ann(ratings_df, embedding_size, epochs):
    res_dict = dict()

    # Encode ratings table to integers
    encoded_data, user_idx2id_dict, course_idx2id_dict = encode_ratings(
        ratings_df)
    # Scale dataset and split it to train/val/test
    X = generate_train_test_datasets_ann(encoded_data)
    x_train, x_val, x_test, y_train, y_val, y_test = X
    # Get size for ANN
    num_users = len(ratings_df['user'].unique())
    num_items = len(ratings_df['item'].unique())
    # Instantiate ANN
    model = RecommenderNet(num_users, num_items, embedding_size)
    model.compile(optimizer=Adam(learning_rate=.003),
                  loss=MeanSquaredError(),
                  metrics=[RootMeanSquaredError()])

    train_me = True
    if train_me:
        run_hist = model.fit(x_train,
                             y_train,
                             validation_data=(x_val, y_val),
                             epochs=epochs,
                             shuffle=True)

    # Evaluate trained ANN
    rmse = model.evaluate(x_test, y_test, verbose=0)

    # Extract embeddings
    # Create a dataframe of the user features
    user_latent_features = model.get_layer(
        'user_embedding_layer').get_weights()[0]
    user_columns = [f"UFeature{i}" for i in range(
        user_latent_features.shape[1])]
    user_embeddings_df = pd.DataFrame(
        data=user_latent_features, columns=user_columns)
    user_embeddings_df['user'] = user_embeddings_df.index
    # Shift column 'user' to first position
    first_column = user_embeddings_df.pop('user')
    user_embeddings_df.insert(0, 'user', first_column)
    # Decode user ids
    user_embeddings_df['user'] = user_embeddings_df['user'].replace(
        user_idx2id_dict)
    # Create a dataframe of the item features
    item_latent_features = model.get_layer(
        'item_embedding_layer').get_weights()[0]
    item_columns = [f"CFeature{i}" for i in range(
        item_latent_features.shape[1])]
    item_embeddings_df = pd.DataFrame(
        data=item_latent_features, columns=item_columns)
    item_embeddings_df['item'] = item_embeddings_df.index
    # Shift column 'item' to first position
    first_column = item_embeddings_df.pop('item')
    item_embeddings_df.insert(0, 'item', first_column)
    # Decode user ids
    item_embeddings_df['item'] = item_embeddings_df['item'].replace(
        course_idx2id_dict)

    # Pack all results
    # res_dict["model"] = model
    res_dict["rmse"] = rmse
    res_dict["user_idx2id_dict"] = user_idx2id_dict
    res_dict["course_idx2id_dict"] = course_idx2id_dict
    res_dict["user_embeddings_df"] = user_embeddings_df
    res_dict["item_embeddings_df"] = item_embeddings_df

    return res_dict, model


def predict_ann_values(model,
                       unselected_course_ids,
                       new_id,
                       training_artifacts):

    result = {}
    # Get and modify mapping dictionaries
    course_idx2id_dict = training_artifacts["course_idx2id_dict"]
    user_idx2id_dict = training_artifacts["user_idx2id_dict"]
    course_id2idx_dict = {v: k for k, v in course_idx2id_dict.items()}
    user_id2idx_dict = {v: k for k, v in user_idx2id_dict.items()}
    # Create dataframe with user data
    courses = list(unselected_course_ids)
    users = [new_id]*len(courses)
    data_dict = {"user": users, "item": courses}
    data_df = pd.DataFrame(data_dict, columns=["user", "item"])
    # Encode data
    data_df['item'] = data_df['item'].map(course_id2idx_dict)
    data_df['user'] = data_df['user'].map(user_id2idx_dict)
    data_df = data_df.dropna()
    # Extract data matrix
    x = data_df[["user", "item"]].values
    # Predict
    pred = model.predict(x)
    # Pack and decode
    data_df["ratings"] = pred.ravel()
    data_df['item'] = data_df['item'].map(course_idx2id_dict)
    data_df['user'] = data_df['user'].map(user_idx2id_dict)
    courses = data_df['item'].to_list()
    ratings = data_df["ratings"].to_list()
    result = {courses[i]: ratings[i] for i in range(len(courses))}

    return result


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    res = {}
    for enrolled_course in enrolled_course_ids:
        for unselected_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselected_course:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselected_course]
                sim = sim_matrix[idx1, idx2]
                if unselected_course not in res:
                    res[unselected_course] = sim
                else:
                    if sim >= res[unselected_course]:
                        res[unselected_course] = sim
    res = {k: v for k, v in sorted(
        res.items(), key=lambda item: item[1], reverse=True)}
    return res

# function for clustering users


def cluster_users(user_profiles_df, pca_variance, num_clusters):
    res_dict = dict()
    feature_names = [f for f in list(
        user_profiles_df.columns) if f not in ['USER_ID']]
    user_ids = user_profiles_df.loc[:, user_profiles_df.columns != 'USER_ID']
    # scale
    scaler = StandardScaler()
    features = user_profiles_df[feature_names]
    res_dict['feature_names'] = feature_names
    features_scaled = scaler.fit_transform(features)
    # PCA
    n_components = len(feature_names)
    if pca_variance < 1:
        pca = PCA(n_components=pca_variance)
    pca = PCA(n_components=n_components)
    features_scaled_pca = pca.fit_transform(features_scaled)
    # k-means clustering
    kmeans = KMeans(n_clusters=num_clusters,
                    random_state=RANDOM_SEED, init='k-means++')
    kmeans.fit(features_scaled_pca)
    # extract cluster labels
    cluster_labels = kmeans.labels_
    # aggregate a dataframe
    labels_df = pd.DataFrame(cluster_labels)
    cluster_df = pd.merge(user_ids, labels_df,
                          left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    res_dict['cluster_df'] = cluster_df

    # add to the pipeline
    pipe = Pipeline([('scaler', scaler),
                     ('pca', pca)
                     ('kmeans', kmeans)])
    res_dict['pipe'] = pipe
    return res_dict


def create_user_profile(enrolled_course_ids, course_genres_df):
    user_profile = np.zeros((1, NUM_GENRES))
    standard_rating = 3
    for enrolled_course in enrolled_course_ids:
        course_descriptor = course_genres_df[course_genres_df.COURSE_ID ==
                                             enrolled_course].iloc[:, 2:].values
        user_profile += standard_rating*course_descriptor

    return user_profile


def compute_user_profile_recommendations(user_profile,
                                         idx_id_dict,
                                         enrolled_course_ids,
                                         course_genres_df):

    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # FIXME: this could be one matrix multiplication
    res = {}
    for unselected_course in unselected_course_ids:
        score = 0.0
        course_descriptor = course_genres_df[course_genres_df.COURSE_ID ==
                                             unselected_course].iloc[:, 2:].values
        score = np.dot(course_descriptor, user_profile.T)[0, 0]
        if unselected_course not in res:
            res[unselected_course] = score
        else:
            if score >= res[unselected_course]:
                res[unselected_course] = score
    res = {k: v for k, v in sorted(
        res.items(), key=lambda item: item[1], reverse=True)}

    return res


def compute_course_user_similarities(rating_sparse_df):
    item_list = rating_sparse_df.column[1:]
    item_list_df = pd.DataFrame(data=item_list, columns=['item'])
    # empty items pairwise
    item_sim = np.zeros((len(item_list), len(item_list)))
    for i, this_item in enumerate(item_list):
        this_item_ratings = rating_sparse_df[this_item].values
        for j, other_item in enumerate(item_list):
            other_item_ratings = rating_sparse_df[other_item].values
            similarity = 1 - cosine(this_item_ratings, other_item_ratings)
          # fixme:
            item_sim[i, j] = similarity
    item_sim_df = pd.DataFrame(data=item_sim, columns=item_list)
    item_sim_df = pd.concat([item_list_df, item_sim_df], axis=1)
    return item_sim_df


def predict_user_clusters(user_profile_df, training_artifacts):
    feature_names = training_artifacts['feature_names']
    pipe = training_artifacts['pipe']
    feature_names = training_artifacts['feature_names']
    clusters = pipe.predict(user_profile_df[feature_names])
    return clusters


def compute_user_cluster_recommendations(cluster, ratings_df, training_artifacts):
    res = {}
    cluster_df = training_artifacts['cluster_df']
    ratings_labelled_df = pd.merge(
        ratings_df, cluster_df, left_on='user', right_on='user')
    courses_clusters = ratings_labelled_df[['item', 'cluster']]
    courses_clusters['count'] = [1] * len(courses_clusters)
    courses_clusters = courses_clusters.groupby(['item', 'cluster']).count().agg(
        enrollments=('count', 'sum')).reset_index()
    recommended_courses = (courses_clusters.loc[courses_clusters.cluster == cluster,
                                                ['item', 'enrollments']]
                           .sort_values(by='enrollments', ascending=False))

    courses = list(recommended_courses.item)
    scores = list(recommended_courses.enrollments)
    res = {courses[i]: scores[i] for i in range(len(courses))}
    return res


def compute_knn_courses(enrolled_course_ids, idx_id_dict, training_artifacts):
    res = {}
    course_sim_df = training_artifacts['course_sim_df']
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # FIXME: this could be one matrix multiplication
    for selected_course in enrolled_course_ids:
        # Get all similarities
        course_sim_row = course_sim_df.loc[course_sim_df.item ==
                                           selected_course]
        for unselected_course in unselected_course_ids:
            score = 0
            if unselected_course in course_sim_row.columns:
                score = course_sim_row[unselected_course].values[0]
            if unselected_course not in res:
                res[unselected_course] = score
            else:
                if score >= res[unselected_course]:
                    res[unselected_course] = score
        res = {k: v for k, v in sorted(
            res.items(), key=lambda item: item[1], reverse=True)}

    return res


def preprocess_embeddings(ratings_df,
                          user_embeddings_df,
                          item_embeddings_df):
    user_emb_merged = pd.merge(ratings_df,
                               user_embeddings_df,
                               how='left',
                               left_on='user',
                               right_on='user').fillna(0)
    # Merge course embedding features
    merged_df = pd.merge(user_emb_merged,
                         item_embeddings_df,
                         how='left',
                         left_on='item',
                         right_on='item').fillna(0)

    # Sum embedding features and create new dataset
    u_feautres = [f"UFeature{i}" for i in range(16)]
    c_features = [f"CFeature{i}" for i in range(16)]

    user_embeddings = merged_df[u_feautres]
    course_embeddings = merged_df[c_features]
    ratings = merged_df['rating']

    # Aggregate the two feature columns using element-wise add
    embedding_dataset = user_embeddings + course_embeddings.values
    embedding_dataset.columns = [f"Feature{i}" for i in range(16)]
    embedding_dataset['rating'] = ratings

    # Extract features and target
    X = embedding_dataset.iloc[:, :-1]
    y = embedding_dataset.iloc[:, -1]

    return X, y


def create_embeddings_frame(user_id, user_embeddings_df, item_embeddings_df):
    # Generate/load data
    ratings_df = load_ratings()
    idx_id_dict, _ = get_doc_dicts()
    user_ratings = ratings_df[ratings_df['user'] == user_id]
    enrolled_course_ids = user_ratings['item'].to_list()
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = list(all_courses.difference(enrolled_course_ids))
    # Create dataframe with courses for which we predict ratings
    ratings_pred_df = pd.DataFrame(unselected_course_ids, columns=['item'])
    ratings_pred_df['user'] = [user_id]*len(unselected_course_ids)
    ratings_pred_df['rating'] = -1
    # Preprocess ratings dataframe
    X, _ = preprocess_embeddings(ratings_pred_df,
                                 user_embeddings_df,
                                 item_embeddings_df)

    return X, unselected_course_ids


def train(model_name, params):
    training_artifacts = {}
    training_artifacts['model_name'] = model_name
    if model_name == models[0]:
        pass
    elif model_name == models[1]:
        pass
    elif model_name == models[2] or model_name == models[3]:
        user_profiles_df = load_user_profiles()
        pca_variances = params['pca_variances']
        res_dict = cluster_users(user_profiles_df=user_profiles_df,
                                 pca_variance=pca_variance, num_clusters=params['num_clusters'])
        training_artifacts.update(res_dict)
    elif model_name == models[4]:
        ratings_df = load_ratings()
        ratings_sparse_df = (ratings_df.pivot(index='user',
                                              columns='item',
                                              values='rating')
                             .fillna(0)
                             .reset_index()
                             .rename_axis(index=None,
                                          columns=None))
        course_sim_df = compute_course_user_similarities(ratings_sparse_df)
        training_artifacts['course_sim_df'] = course_sim_df
    elif model_name == models[5]:
        ratings_df = load_ratings()
        ratings_sparse_df = (ratings_df.pivot(index='user',
                                              columns='item',
                                              values='rating')
                             .fillna(0)
                             .reset_index()
                             .rename_axis(index=None,
                                          columns=None))
        num_components = params['num_components']
        nmf = NMF(n_components=num_components,
                  init='random',
                  random_state=RANDOM_SEED)
        nmf = nmf.fit(ratings_sparse_df.iloc[:, 1:])
        compo = nmf.components_
        training_artifacts['components'] = compo
        training_artifacts['nmf'] = nmf
    elif model_name == models[6]:
        ratings_df = load_ratings()
        # Extract user parameters
        num_components = params['num_components']
        num_epochs = params['num_epochs']
        # Train ANN
        res_dict, model = train_ann(ratings_df, num_components, num_epochs)
        # Extend training_artifacts with the new created elements from res_dict
        training_artifacts.update(res_dict)
        # FIXME:
        # Tensorflow models cannot be passed in a dict,
        # because they're not hashable.
        # A quick and dirty solution is to compute the prediction here...
        # training_artifacts["model"] = model
        new_id = params["new_id"]
        idx_id_dict, _ = get_doc_dicts()
        all_courses = set(idx_id_dict.values())
        user_ratings = ratings_df[ratings_df['user'] == new_id]
        enrolled_course_ids = user_ratings['item'].to_list()
        unselected_course_ids = all_courses.difference(enrolled_course_ids)
        result = predict_ann_values(model,
                                    unselected_course_ids,
                                    new_id,
                                    training_artifacts)
        training_artifacts["result"] = result
        # Prepare inputs for sub-options: regression & classification with embeddings
        user_embeddings_df = training_artifacts["user_embeddings_df"]
        item_embeddings_df = training_artifacts["item_embeddings_df"]
        X, y = preprocess_embeddings(ratings_df,
                                     user_embeddings_df,
                                     item_embeddings_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X,  # predictive variables
            y,  # target
            test_size=0.1,  # portion of dataset to allocate to test set
            random_state=RANDOM_SEED  # we are setting the seed here, ALWAYS DO IT!
        )


def predict(model_name, user_ids, params, training_artifacts):
    sim_threshold = 0.6
    if 'sim_threshold' in params:
        sim_threshold = params['sim_threshold']/100
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = dict()
    score_threshold = -1
    score_description = ''
    try:
        assert "model_name" in training_artifacts
    except AssertionError as error:
        raise (error)
    for user_id in user_ids:
        if model_name == models[0]:
            sim_threshold = .5
            if 'sim_threshold' in params:
                sim_threshold = params['sim_threshold'] / 100
            score_threshold = sim_threshold
            # load data
            idx_id_dict, id_idx_dict = get_doc_dicts()
            sim_matrix = load_course_sims().to_numpy()
            ratings_df = load_ratings()
            # predict
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict,
                                                    id_idx_dict,
                                                    enrolled_course_ids,
                                                    sim_matrix)
            score_description = "Note: the score is the cosine similarity\
                                 between the selected and the recommended\
                                 courses."

        elif model_name == models[1]:
            profile_threshold = 0
            if 'profile_threshold' in params:
                profile_threshold = params['profile_threshold']
            score_threshold = profile_threshold
            # load data
            course_genres_df = load_course_genres
            idx_id_dict, _ = get_doc_dicts()
            ratings_df = load_ratings()

            # Create user profile vector: (1,14)
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            user_profile = create_user_profile(enrolled_course_ids,
                                               course_genres_df)
            # Predict
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = compute_user_profile_recommendations(user_profile,
                                                       idx_id_dict,
                                                       enrolled_course_ids,
                                                       course_genres_df)
            score_description = "Note: the score is the alignment (dot product)\
                                 between the user profile built with the selected\
                                 courses and the recommended ones."

        elif model_name == models[2] or model_name == models[3]:
            # load data
            course_genres_df = load_course_genres()
            idx_id_dict, _ = get_doc_dicts()
            ratings_df = load_ratings()
            # create user profile vector
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            user_profile = create_user_profile(enrolled_course_ids,
                                               course_genres_df)
            user_profile_df = pd.DataFrame(data=user_profile,
                                           columns=course_genres_df.columns[2:])
            # predict
            cluster = predict_user_clusters(
                user_profile_df, training_artifacts)[0]
            res = compute_user_cluster_recommendations(
                cluster, ratings_df, training_artifacts)
            score_description = "Note: the score is the number of enrollments\
                of each recommended course, which belongs to the user\
                cluster of the interacting user."
        elif model_name == models[4]:
            idx, _ = get_doc_dicts()
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()

            res = compute_knn_courses(
                enrolled_course_ids, idx_id_dict, training_artifacts)
            score_description = "Note: the score is the cosine similarity\
                of the suggested course with respect to one\
                of the selected courses."
        elif model_name == models[5]:
            ratings_df = load_ratings()
            # FIXME: should not make sparse the complete frame, but only the user rows
            ratings_sparse_df = (ratings_df.pivot(index='user',
                                                  columns='item',
                                                  values='rating')
                                 .fillna(0)
                                 .reset_index()
                                 .rename_axis(index=None,
                                              columns=None))
            user_ratings_sparse = ratings_sparse_df[ratings_sparse_df['user'] == user_id]
            H = training_artifacts['components']
            nmf = training_artifacts['nmf']
            W = nmf.transform(user_ratings_sparse.iloc[:, 1:])
            X_hat = np.dot(W, H)
            items = list(ratings_sparse_df.columns[1:])
            ratings = list(X_hat.ravel())
            res = {items[i]: ratings[i] for i in range(len(items))}
            score_description = "Note: the score is the rating predicted\
                by the Non-Negative Matrix Factorization (NMF) model."
        elif model_name == models[6]:
            res = training_artifacts['result']
            score_description = "Note: the score is the rating predicted\
                by the neural network model."
    # filter the results
    for key, score in res.items():
        if score >= score_threshold:
            users.append(user_id)
            courses.append(key)
            scores.append(score)
    # create dataframe with results
    res_dict['USER'] = users
    res_dict['COURSE'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE', 'SCORE'])
    res_df = res_df.drop_duplicates(
        subset=['COURSE_ID']).reset_index(drop=True)
    # Restrict number of results, if required
    if "top_courses" in params:
        top_courses = params["top_courses"]
        if res_df.shape[0] > top_courses and top_courses > 0:
            # Sort according to score
            res_df.sort_values(by='SCORE', ascending=False, inplace=True)
            # Select top_courses
            res_df = res_df.reset_index(drop=True).iloc[:top_courses, :]

    return res_df, score_description
