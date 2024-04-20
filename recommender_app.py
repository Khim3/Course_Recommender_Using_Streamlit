import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
    page_title='Course Recommender System',
    layout='wide',
    initial_sidebar_state='expanded'
)
# Functions
# load datasets

def load_ratings():
    return backend.load_ratings()


def load_course_sims():
    return backend.load_course_sims()


def load_courses():
    return backend.load_courses()


def load_bow():
    return backend.load_bow()
# Initialize the app by first loading datasets

def init__recommender_app():
    with st.spinner("Loading datasets "):
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
    for i in range(7):
        if model_name == backend.models[i]:
            with st.spinner('Training'):
                time.sleep(0.5)
                backend.train(model_name, params)
            st.success('Model trained successfully')
    # TODO: add other model


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
if model_selection == backend.models[0]:
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=1,
                                    max_value=100,
                                    value=10, step=1)
    course_sim_threshold = st.sidebar.slider('Course Similarity %',
                                             min_value=0,
                                             max_value=100,
                                             value=50, step=5)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
elif model_selection == backend.models[1]:  # user profile
    num_clusters = st.sidebar.slider('Number of Clusters',
                                     min_value=2,
                                     max_value=30,
                                     value=5, step=1)
    params['num_clusters'] = num_clusters
   # add hyperparameter for this
elif model_selection == backend.models[2]:  # clustering kmeans
    num_clusters = st.sidebar.slider('Number of Clusters',
                                     min_value=2,
                                     max_value=30,
                                     value=5, step=1)
    params['num_clusters'] = num_clusters
elif model_selection == backend.models[3]:  # clustering with pca
    num_components = st.sidebar.slider('Number of Components',
                                       min_value=2,
                                       max_value=30,
                                       value=5, step=1)
    params['num_components'] = num_components
    # TODO: add hyper-parameters for this
elif model_selection == backend.models[4]:  # knn
    k_neighbors = st.sidebar.slider('Number of Neighbors',
                                    min_value=1,
                                    max_value=20,
                                    value=5, step=1)
    params['k_neighbors'] = k_neighbors
    # TODO: add hyper-parameters for this
elif model_selection == backend.models[5]:  # NMF
    num_factors = st.sidebar.slider(
        'Number of factors', min_value=1, max_value=100, value=10, step=1)
    params['num_factors'] = num_factors
    # TODO: add hyper-parameters for this

elif model_selection == backend.models[6]:  # Neural network
    pass
    # TODO: add hyper-parameters for this

# TODO: Add hyper-parameters for other models
else:
    pass

# training
st.sidebar.subheader('3. Training')
training_button = st.sidebar.button('Train Model')
train_text = st.sidebar.text('')
# start training process
if training_button:
    backend.train(model_selection, params)
    st.sidebar.success('Model trained successfully')
# prediction
st.sidebar.subheader('4. Prediction')
# start prediction process
prediction_button = st.sidebar.button('Generate Recommendations')

if prediction_button and selected_courses_df.shape[0] > 0:
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = backend.predict(model_selection, user_ids, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on='COURSE_ID')
    st.table(res_df)
