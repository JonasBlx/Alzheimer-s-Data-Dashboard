import streamlit as st

def display_patient_profile(data, patient_id):
    """
    Display the profile of a patient based on their ID.
    - Shows a table of the patient's details.
    """
    patient = data[data['PatientID'] == patient_id]

    if patient.empty:
        st.warning("No patient found with this ID.")
    else:
        st.subheader(f"Patient Details for ID {patient_id}")

        # Display patient information
        patient_info = patient.transpose()
        patient_info.columns = ['Value']
        st.table(patient_info)
