import streamlit as st

def show_tab():
    st.title("About PacCEM")
    st.markdown("""
    Welcome to PacCEM (Pacific Capacity Expansion Model), a user-friendly tool designed to simplify power system modeling for small and weak grids.
    This application empowers utilities, regulators, researchers, and students to perform transparent, reproducible, and offline-capable capacity expansion and dispatch analyses without requiring deep programming knowledge.
    """)

    st.subheader("How to Use")
    st.markdown("""
    1.  **Project Tab:** Start by defining your project name, results directory, and scenario-specific parameters like solver, CO2 cap, RE share, and technology cost multipliers. You will also upload your main Excel data file here.
    2.  **Data Mapping Tab:** Map the sheets and columns from your uploaded Excel file to the required PyPSA components. Alternatively, you can manually enter data directly into interactive tables.
    3.  **Simulation Tab:** Once all data is mapped and saved, initiate the simulation. You can monitor the progress through a live log feed and visualize network buses on an interactive map. Download comprehensive results after completion.
    4.  **Compare Tab:** Upload results from single or multiple scenarios (CSV files) to generate interactive plots for comparative analysis.
    """)

    st.subheader("Built with")
    st.markdown("""
    *   **Frontend:** Streamlit (Python web framework)
    *   **Backend:** PyPSA (Python for Power System Analysis)
    *   **Data Persistence:** SQLite (per-project)
    *   **Interoperability:** YAML (for scenario configuration), Excel/CSV (for inputs), NetCDF/CSV/PDF (for outputs)
    *   **Solvers:** HiGHS, CBC, Gurobi (user-selectable)
    """)

    st.subheader("Requirements")
    st.markdown("""
    *   Python 3.8+
    *   Required Python packages (listed in `requirements.txt`)
    *   An installed PyPSA-compatible solver (e.g., HiGHS, CBC, Gurobi)
    """)

    st.subheader("Builder Information")
    st.info("Developed by a dedicated team to democratize access to power system modeling.")

    # Navigation buttons (only Next for the first tab)
    col1, col2 = st.columns([1, 10])
    with col2:
        if st.button("Next >", key="about_next"):
            st.session_state.current_tab_index = 1
            st.rerun()