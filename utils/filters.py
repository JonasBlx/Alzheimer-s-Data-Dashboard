import streamlit as st


def add_filters(data):
    """
    Add sidebar filters for interactive data exploration.
    - Filters include age range, gender, and diagnosis.
    """
    st.sidebar.header("🔎 Filters")

    # Age filter
    age_min = int(data['Age'].min())
    age_max = int(data['Age'].max())
    age_range = st.sidebar.slider(
        "Age Range",
        age_min, age_max,
        (age_min, age_max)
    )

    # Gender filter
    gender_options = {'All': 'All', 'Male': 0, 'Female': 1}
    selected_gender = st.sidebar.selectbox(
        "Gender",
        options=list(gender_options.keys())
    )
    gender_value = gender_options[selected_gender]

    # Diagnosis filter
    diagnosis_options = ['All'] + data['Diagnosis'].unique().tolist()
    selected_diagnosis = st.sidebar.selectbox(
        "Diagnosis", options=diagnosis_options
        )

    # Apply filters
    filtered_data = data[
        (data['Age'] >= age_range[0]) &
        (data['Age'] <= age_range[1])
    ]

    if gender_value != 'All':
        filtered_data = filtered_data[filtered_data['Gender'] == gender_value]

    if selected_diagnosis != 'All':
        filtered_data = filtered_data[
            filtered_data['Diagnosis'] == selected_diagnosis
            ]

    return filtered_data
