import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

ALLOW_ANN = False
st.set_page_config(
    page_title='Course Recommender System',
    layout='wide',
    initial_sidebar_state='expanded'
)
# Functions
# load datasets


@st.cache_data
def load_ratings():
    return backend.load_ratings()


def load_course_sims():
    return backend.load_course_sims()


def load_courses():
    return backend.load_courses()


def load_bow():
    return backend.load_bow()


def load_course_genres():
    return backend.load_course_genres()


def load_user_profiles():
    return backend.load_user_profiles()
# Initialize the app by first loading datasets


def init__recommender_app():
    with st.spinner("Loading datasets"):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # select courses
    st.success('Dataset loaded successfully')
    st.markdown('---')
    st.subheader('Select courses that you have audited or completed: ')

    # interactive table for course df
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(
        enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode='multiple', use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )
    results = pd.DataFrame(response["selected_rows"], columns=[
                           'COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader('YOUR COURSES:')
    st.table(results)
    return results


def train(model_name, params):
    training_artifacts = None
    try:
        assert model_name in backend.models
        model_index = backend.get_model_index(model_selection)
        do_train = False
        if model_index > 5:
            if ALLOW_ANN:
                do_train = True
        else:
            do_train = True
        if do_train:
            with st.spinner('Training model...'):
                time.sleep(0.75)
                training_artifacts = backend.train(model_name, params)
            st.success('Model trained successfully!!!')
        else:
            st.write('Training not allowed for this model')
        return training_artifacts
    except Exception as e:
        st.error(f'Error training model: {e}')
        raise e
def predict(model_name, params, training_artifacts):
    res = None
    try:
        assert model_name in backend.models
        model_index = backend.get_model_index(model_selection)
        do_predict = False
        if model_index > 5: # Neural Networks & Co.
            if ALLOW_ANN:
                do_predict = True            
        else:
            do_predict = True
        if do_predict:
            with st.spinner('Generating course recommendations: '):
                time.sleep(0.5)
                # FIXME: new_id is also contained in params to trigger the cache miss for train()
                # That makes user_ids redundant, however, necessary if we want to
                # - maintain a lower API for predict()
                
                new_id = params["new_id"]
                user_ids = [new_id]
                res, descr = backend.predict(model_name, user_ids, params, training_artifacts)
            st.success('Recommendations generated!')
            st.write(f"**{backend.models[model_index][3:]}**: {backend.MODEL_DESCRIPTIONS[model_index]}")          
            st.write(descr)
        else:
            st.write("Sorry, the Neural Networks model is not active at the moment\
                due to the slug memory quota on Heroku. \
                If you clone the repository, \
                you can try it on your local machine, though.")
        return res
    except AssertionError as err:
        print("Model name must be in the drop down.") # we should use the logger
        raise err

# ----UI-----
# sidebar
st.sidebar.title('Course Learning Recommender')
# initialize app
selected_courses_df = init__recommender_app()
# selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    'Select model:',
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Hyper-parameters tuning: ')
top_courses = st.sidebar.slider('Top courses',
                                min_value=1, max_value=100,
                                value=10, step=1)
params['top_courses'] = top_courses
if model_selection == backend.models[0]:  
    course_sim_threshold = st.sidebar.slider('Course Similarity %',
                                             min_value=0,
                                             max_value=100,
                                             value=50, step=5)
    
    params['sim_threshold'] = course_sim_threshold
elif model_selection == backend.models[1]:  # user profile
    profile_threshold = st.sidebar.slider('Alignment Scores',
                                          min_value=0,
                                          max_value=100,
                                          value=50, step=1)
    params['profile_threshold'] = profile_threshold
   # add hyperparameter for this
elif model_selection == backend.models[2]:  # clustering kmeans
    num_clusters = st.sidebar.slider('Number of Clusters',
                                     min_value=2,
                                     max_value=30,
                                     value=5, step=1)
    params['num_clusters'] = num_clusters
    params['pca_variance'] = 1
elif model_selection == backend.models[3]:  # clustering with pca
    num_clusters = st.sidebar.slider('Number of clusters',
                                     min_value=1, max_value=30,
                                     value=11, step=1)
    pca_variance = st.sidebar.slider('Genre variance coverage (PCA)',
                                     min_value=0, max_value=100,
                                     value=90, step=5)
    params['num_clusters'] = num_clusters
    params['pca_variance'] = pca_variance / 100
    # TODO: add hyper-parameters for this
elif model_selection == backend.models[4]:  # knn
    pass
elif model_selection == backend.models[5]:  # NMF
    num_factors = st.sidebar.slider(
        'Number of factors', min_value=1, max_value=50, value=10, step=1)
    params['num_factors'] = num_factors
  

elif model_selection == backend.models[6]:  # Neural network
    num_components = st.sidebar.slider('Number of latent components (embedding size)',
                                   min_value=1, max_value=30,
                                   value=16, step=1)
    num_epochs = st.sidebar.slider('Number of epochs',
                                   min_value=1, max_value=10,
                                   value=1, step=1)
    params['num_components'] = num_components
    params['num_epochs'] = num_epochs
else:
    pass

# training
st.sidebar.subheader('3. Training')
training_button =False
training_button = st.sidebar.button('Train Model')
train_text = st.sidebar.text('')
training_artifacts = None
model_index = backend.get_model_index(model_selection)
# start training process
if training_button:
    training_artifacts = train(model_selection, params)
    st.sidebar.success('Model trained successfully')

# prediction
st.sidebar.subheader('4. Prediction')
# start prediction process
prediction_button = st.sidebar.button('Generate Recommendations')

if prediction_button and selected_courses_df.shape[0] > 0:
    if model_index < 6 and not training_artifacts:
        # Since train() is cached, we don't really recompute everything
        # All models which are not based on ANN embeddings don't have
        # the new user entries in the ratings dataset yet - we need to add them now.
        training_artifacts = train(model_selection, params)
    # Create a new id for current user session
    # We create a new entry in the ratings.csv for the interactive user
    # who has selected the courses in the UI
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    params["new_id"] = new_id

    if new_id:
        res_df = predict(model_selection, params, training_artifacts)
        res_df = res_df[['COURSE_ID', 'SCORE']]
        course_df = load_courses()
        res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
        st.table(res_df)
