import streamlit as st
import pandas as pd
import io
import ast # For literal_eval

# Helper function to read a sheet and populate dropdowns
def get_sheet_and_columns(sheet_name, excel_file_buffer):
    if not excel_file_buffer or not sheet_name:
        return None, []
    try:
        df = pd.read_excel(io.BytesIO(excel_file_buffer), sheet_name=sheet_name)
        return df, df.columns.tolist()
    except Exception as e:
        st.error(f"Error reading sheet '{sheet_name}': {e}")
        return None, []

# Helper to render dropdown for column selection
def column_selector(component_type, attribute_name, columns, default_value=None):
    key = f"{component_type}_{attribute_name}_col"
    st.session_state.mapped_data[component_type] = st.session_state.mapped_data.get(component_type, {})
    
    current_value = st.session_state.mapped_data[component_type].get(attribute_name, default_value)
    
    if columns:
        selected_col = st.selectbox(
            f"{attribute_name} Column",
            options=["-- Select --"] + columns,
            index=columns.index(current_value) + 1 if current_value in columns else 0,
            key=key,
            help=f"Select the column from the Excel sheet that contains {attribute_name}."
        )
        if selected_col != "-- Select --":
            st.session_state.mapped_data[component_type][attribute_name] = selected_col
        else:
            st.session_state.mapped_data[component_type].pop(attribute_name, None) # Remove if unselected
            
    else:
        st.warning(f"No columns available for {component_type}. Please select a sheet first.")
        st.session_state.mapped_data[component_type].pop(attribute_name, None) # Ensure no mapping if no columns

# Helper to render data editor for manual entry
def manual_data_editor(component_type, default_cols):
    st.markdown(f"**Manually Enter {component_type.replace('_', ' ').title()} Data**")
    
    # Initialize DataFrame in session_state if not present or columns changed
    if component_type not in st.session_state.manual_data or \
       list(st.session_state.manual_data[component_type].columns) != default_cols:
        st.session_state.manual_data[component_type] = pd.DataFrame(columns=default_cols)

    edited_df = st.data_editor(
        st.session_state.manual_data[component_type],
        num_rows="dynamic",
        key=f"manual_{component_type}_editor"
    )
    st.session_state.manual_data[component_type] = edited_df
    

def show_tab():
    st.title("Data Mapping & Manual Entry")

    if not st.session_state.get('excel_file_buffer'):
        st.warning("Please upload an Excel file in the 'Project' tab to enable Excel mapping.")
        return

    excel_sheet_names = st.session_state.get('excel_sheet_names', [])
    if not excel_sheet_names:
        st.warning("No sheets found in the uploaded Excel file. Please check the file.")
        return

    if 'data_mapping_mode' not in st.session_state:
        st.session_state.data_mapping_mode = {}
    
    # CURRENCY: Updated column names to USD
    component_types_spec = {
        "buses": ['Bus name', 'V_nom', 'x', 'y', 'Carriers'],
        "demand": [],
        "generators": ['Generator name', 'Bus', 'Size (MW)', 'Quantity', 'Build Year', 'Capacity(MW)', 'P_nom_min', 'P_nom_max', 'Carrier', 'Scenario', 'p_nom_extendable', 'Variable cost (USD/MWh)', 'Capital_cost (USD/MW)', 'lifetime', 'Status', 'efficiency', 'min_generation_level'],
        "transmission_lines": ['From', 'To', 'type', 's_nom_extendable', 's_nom', 'Capital_cost (USD/MW/km)', 'Length (kM)'],
        "transformers": ['Location', 'bus0', 'bus1', 's_nom', 'v_nom0', 'v_nom1', 'x', 'r', 'Capital_cost (USD)'],
        "storage": ['name', 'Capacity(MW)', 'Year', 'Carrier', 'Bus', 'Scenario', 'e_nom_extendable', 'Variable cost (USD/MWh)', 'Capital_cost (USD/MWh)', 'lifetime', 'Status'],
        "generation_profiles": ['Solar profile', 'Wind profile', 'Hydro profile']
    }

    sub_tab_names = list(component_types_spec.keys())
    sub_tabs = st.tabs([name.replace('_', ' ').title() for name in sub_tab_names])

    for i, component_type in enumerate(sub_tab_names):
        with sub_tabs[i]:
            st.subheader(f"{component_type.replace('_', ' ').title()} Data")
            
            st.session_state.data_mapping_mode[component_type] = st.radio(
                f"Select data input mode for {component_type.replace('_', ' ').title()}",
                ("Excel Mapping", "Manual Entry"),
                key=f"{component_type}_mode_radio",
                index=0 if st.session_state.data_mapping_mode.get(component_type, "Excel Mapping") == "Excel Mapping" else 1,
            )

            if st.session_state.data_mapping_mode[component_type] == "Excel Mapping":
                st.session_state.mapped_data[component_type] = st.session_state.mapped_data.get(component_type, {})
                selected_sheet_key = f"{component_type}_sheet_selector"
                
                selected_sheet = st.selectbox(
                    f"Select Excel Sheet for {component_type.replace('_', ' ').title()}",
                    options=["-- Select --"] + excel_sheet_names,
                    index=excel_sheet_names.index(st.session_state.mapped_data[component_type].get('sheet_name', '')) + 1
                          if st.session_state.mapped_data[component_type].get('sheet_name') in excel_sheet_names else 0,
                    key=selected_sheet_key
                )
                
                if selected_sheet != "-- Select --":
                    st.session_state.mapped_data[component_type]['sheet_name'] = selected_sheet
                    df_current_sheet, current_sheet_cols = get_sheet_and_columns(selected_sheet, st.session_state.excel_file_buffer)
                    if df_current_sheet is None:
                        continue

                    if component_type == "demand":
                        st.info("For Demand, all columns except the index column (if any) will be considered load profiles for different buses.")
                        if df_current_sheet is not None:
                            st.dataframe(df_current_sheet.head())
                            st.session_state.mapped_data[component_type]['df_content'] = df_current_sheet.to_dict('list')
                    elif component_type == "generation_profiles":
                        enabled_techs = st.session_state.project_data.get('enabled_techs', {})
                        st.info("Select profile columns for enabled renewable technologies.")
                        if df_current_sheet is not None:
                            st.dataframe(df_current_sheet.head())
                            for profile_col in ['Solar profile', 'Wind profile', 'Hydro profile']:
                                tech_name = profile_col.split(' ')[0]
                                if enabled_techs.get(tech_name, False):
                                    column_selector(component_type, profile_col, current_sheet_cols)
                                else:
                                    st.session_state.mapped_data[component_type].pop(profile_col, None)
                                    st.info(f"{tech_name} is disabled in Project tab, skipping profile mapping.")
                            st.session_state.mapped_data[component_type]['df_content'] = df_current_sheet.to_dict('list')
                    else:
                        st.dataframe(df_current_sheet.head())
                        for col_name in component_types_spec[component_type]: # Use the spec for column names
                            column_selector(component_type, col_name, current_sheet_cols)
                        st.session_state.mapped_data[component_type]['df_content'] = df_current_sheet.to_dict('list')
                else:
                    st.session_state.mapped_data[component_type].pop('sheet_name', None)
                    st.session_state.mapped_data[component_type].pop('df_content', None)
                    st.warning("Please select a sheet to proceed with mapping.")
            
            else: # Manual Entry
                # Use the spec for manual data columns
                manual_data_editor(component_type, component_types_spec[component_type] if component_type != "demand" else ['Bus'] + [f'Time_{i}' for i in range(8760)])
                if component_type == "generation_profiles":
                    st.warning("For manual generation profiles, ensure you enter 8760 hourly values for each enabled technology.")


            if st.button(f"Save {component_type.replace('_', ' ').title()} Data", key=f"save_data_{component_type}"):
                st.success(f"{component_type.replace('_', ' ').title()} data saved for current session.")

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Next >", key="data_mapping_next"):
            is_valid = False
            for comp_type in component_types_spec.keys():
                if st.session_state.data_mapping_mode[comp_type] == "Excel Mapping":
                    if st.session_state.mapped_data.get(comp_type, {}).get('df_content') is not None:
                        is_valid = True
                        break
                else:
                    if not st.session_state.manual_data.get(comp_type, pd.DataFrame()).empty:
                        is_valid = True
                        break
            
            if not is_valid:
                st.error("Please provide data for at least one component (e.g., Buses) either by Excel mapping or manual entry.")
            else:
                st.session_state.current_tab_index = 3
                st.rerun()
    with col1:
        if st.button("< Back", key="data_mapping_back"):
            st.session_state.current_tab_index = 1
            st.rerun()