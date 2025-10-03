import streamlit as st
import pandas as pd
import os
import io
import plotly.express as px
import plotly.graph_objects as go
from backend.model_runner import load_network_from_nc, extract_key_metrics, get_time_series_component_info, extract_selected_time_series, get_renewable_carriers


def show_tab():
    st.title("Compare Simulation Scenarios (.nc Files)")

    st.sidebar.header("Comparison Controls")

    # Initialize session state for loaded networks
    if 'comparison_networks' not in st.session_state:
        st.session_state.comparison_networks = {} # Stores {scenario_name: pypsa.Network object}
    if 'comparison_uploaded_file_info' not in st.session_state:
        # Stores {original_filename: {'scenario_name': user_given_name, 'file_hash': hash_of_content}}
        st.session_state.comparison_uploaded_file_info = {}
    
    # --- File Upload Section ---
    st.subheader("Upload NetCDF (.nc) Result Files")
    current_uploaded_files = st.file_uploader(
        "Upload one or more PyPSA NetCDF result files (.nc) to compare.",
        type=["nc"],
        accept_multiple_files=True,
        key="nc_file_uploader",
        help="These files are generated after a successful simulation run."
    )

    # Dictionary to hold info about files currently in the uploader, keyed by original filename
    current_uploader_file_map = {f.name: f for f in current_uploaded_files}

    # Identify files removed from the uploader and clear their state
    files_removed_from_uploader = [
        filename for filename in st.session_state.comparison_uploaded_file_info.keys()
        if filename not in current_uploader_file_map
    ]
    for filename in files_removed_from_uploader:
        scenario_name_to_remove = st.session_state.comparison_uploaded_file_info[filename]['scenario_name']
        if scenario_name_to_remove in st.session_state.comparison_networks:
            del st.session_state.comparison_networks[scenario_name_to_remove]
        del st.session_state.comparison_uploaded_file_info[filename]
        # Clean up the text_input's key from session_state too
        if f"scenario_name_input_{filename}" in st.session_state:
            del st.session_state[f"scenario_name_input_{filename}"]
        st.info(f"Removed '{scenario_name_to_remove}' (original: {filename}) from loaded networks.")
    
    # Process newly added or re-uploaded files
    for uploaded_file in current_uploaded_files:
        original_filename = uploaded_file.name
        file_name_base = os.path.splitext(original_filename)[0]
        file_content = uploaded_file.getvalue()
        file_hash = hash(file_content) # Simple hash to detect content changes

        # State for this specific uploaded file
        existing_file_info = st.session_state.comparison_uploaded_file_info.get(original_filename)

        scenario_input_key = f"scenario_name_input_{original_filename}"

        # Initialize the value for the text_input in session state if it doesn't exist
        if scenario_input_key not in st.session_state:
            st.session_state[scenario_input_key] = existing_file_info['scenario_name'] if existing_file_info else file_name_base
        
        # Instantiate the text_input widget
        scenario_name_from_input = st.text_input(
            f"Scenario Name for '{original_filename}'",
            value=st.session_state[scenario_input_key], # Read from session state
            key=scenario_input_key # Streamlit manages updates to st.session_state[key]
        )
        # We no longer manually assign to st.session_state[scenario_input_key] here.
        # The widget's 'value' parameter and 'key' handle it.


        needs_reloading = True
        if existing_file_info and existing_file_info['file_hash'] == file_hash and \
           existing_file_info['scenario_name'] == scenario_name_from_input:
            # File already processed, content and user-given scenario name haven't changed
            needs_reloading = False
            # Check for silent renaming if user re-used a name. If so, rerun is needed.
            if existing_file_info['scenario_name'] != scenario_name_from_input:
                 st.rerun() # Rerun to update comparison_networks keys

        if not needs_reloading:
            # If name changed for an already loaded file, handle dictionary key update
            old_scenario_name = existing_file_info['scenario_name']
            if scenario_name_from_input != old_scenario_name:
                if old_scenario_name in st.session_state.comparison_networks:
                    net_obj = st.session_state.comparison_networks.pop(old_scenario_name)
                    st.session_state.comparison_networks[scenario_name_from_input] = net_obj
                st.session_state.comparison_uploaded_file_info[original_filename]['scenario_name'] = scenario_name_from_input
                st.info(f"Scenario name for '{original_filename}' updated to '{scenario_name_from_input}'.")
                st.rerun()
            continue # If no reloading needed and name handled, move to next uploaded_file


        # If we reach here, 'needs_reloading' is True (new file or content/scenario name changed, or collision)
        
        # Check for name collision before loading (critical for new files or new names)
        name_collision_detected = False
        for s_name_in_state, n_obj_in_state in st.session_state.comparison_networks.items():
            if s_name_in_state == scenario_name_from_input:
                # If name matches an already loaded network, AND it's not the current file being refreshed,
                # then it's a collision.
                # Identify if the existing network with this name came from a DIFFERENT original file.
                original_file_for_colliding_name = None
                for k, v in st.session_state.comparison_uploaded_file_info.items():
                    if v['scenario_name'] == s_name_in_state:
                        original_file_for_colliding_name = k
                        break
                
                # Collision if it's not the same file (meaning another distinct file already uses this name)
                if original_file_for_colliding_name != original_filename:
                    name_collision_detected = True
                    st.warning(f"Scenario name '{scenario_name_from_input}' is already used by '{original_file_for_colliding_name}'. Please choose a unique name for '{original_filename}'.")
                    break

        if name_collision_detected:
            continue # Skip loading this file due to collision


        # Proceed to load the network
        try:
            temp_nc_path = os.path.join("cache", f"temp_{original_filename}")
            with open(temp_nc_path, "wb") as f:
                f.write(file_content)
            
            n = load_network_from_nc(temp_nc_path)
            st.session_state.comparison_networks[scenario_name_from_input] = n
            st.session_state.comparison_uploaded_file_info[original_filename] = {
                'scenario_name': scenario_name_from_input,
                'file_hash': file_hash
            }
            st.success(f"Loaded '{scenario_name_from_input}' from '{original_filename}' successfully.")
            os.remove(temp_nc_path)
            st.rerun() # Rerun to update the selector widgets and process next file cleanly

        except ValueError as e:
            st.error(f"Error loading '{original_filename}': {e}. Please ensure it's a valid PyPSA NetCDF file.")
        except Exception as e:
            st.error(f"An unexpected error occurred loading '{original_filename}': {e}")
        
    # --- Clear All Loaded Networks Button ---
    if st.session_state.comparison_networks:
        if st.button("Clear All Loaded Networks", key="clear_networks"):
            st.session_state.comparison_networks = {}
            st.session_state.comparison_uploaded_file_info = {}
            # Clear all text_input keys from session state
            for key in list(st.session_state.keys()):
                if key.startswith("scenario_name_input_"):
                    del st.session_state[key]
            st.success("All loaded networks cleared.")
            st.rerun()


    # --- Main Comparison Display ---
    if not st.session_state.comparison_networks:
        st.info("Please upload NetCDF files to start comparison.")
        return # Exit if no networks are loaded

    st.subheader("Select Scenarios for Comparison")
    all_scenario_names_in_state = list(st.session_state.comparison_networks.keys())
    
    selected_scenarios = st.sidebar.multiselect(
        "Choose scenarios to compare:",
        options=all_scenario_names_in_state,
        default=all_scenario_names_in_state,
        key="comparison_scenario_selector"
    )
    
    if not selected_scenarios:
        st.warning("Please select at least one scenario to compare.")
        return # Exit if no scenarios are selected for comparison

    networks_to_compare = {s_name: st.session_state.comparison_networks[s_name] for s_name in selected_scenarios}
    
    # --- Numerical Comparison ---
    st.subheader("Numerical Comparison of Key Metrics")
    
    all_metrics_data = []
    for s_name, n_obj in networks_to_compare.items():
        metrics = extract_key_metrics(n_obj, s_name)
        all_metrics_data.append(metrics)
    
    df_metrics_raw = pd.DataFrame(all_metrics_data).set_index('Scenario')

    available_scalar_metrics = [col for col in df_metrics_raw.columns if col not in ['Scenario']]
    if available_scalar_metrics:
        selected_scalar_metrics = st.sidebar.multiselect(
            "Select scalar metrics to display:",
            options=available_scalar_metrics,
            default=available_scalar_metrics[:min(5, len(available_scalar_metrics))],
            key="scalar_metric_selector"
        )
        if selected_scalar_metrics:
            st.dataframe(df_metrics_raw[selected_scalar_metrics].T.round(2))
        else:
            st.info("No scalar metrics selected for display.")
    else:
        st.info("No scalar metrics available for comparison.")

    st.markdown("---")

    # --- Time-Series Plotting ---
    st.subheader("Time-Series Data Comparison")

    combined_ts_info = {} # {ComponentType: {'df_attr_prefix': ..., 'attributes': [...]}}
    for n_obj in networks_to_compare.values():
        current_network_ts_info = get_time_series_component_info(n_obj)
        for comp_type, info in current_network_ts_info.items():
            if comp_type not in combined_ts_info:
                combined_ts_info[comp_type] = info.copy()
            else:
                combined_ts_info[comp_type]['attributes'] = sorted(list(set(combined_ts_info[comp_type]['attributes']).union(set(info['attributes']))))
    
    time_series_component_types = sorted(list(combined_ts_info.keys()))

    if not time_series_component_types:
        st.info("No time-series data components found in the loaded networks. Ensure the .nc files contain time-series data (e.g., generators_t.p, loads_t.p_set).")
        return

    selected_component_type = st.sidebar.selectbox(
        "Select Component Type:",
        options=time_series_component_types,
        key="ts_component_type_selector"
    )

    if selected_component_type:
        available_attributes = combined_ts_info.get(selected_component_type, {}).get('attributes', [])
        
        if not available_attributes:
            st.info(f"No time-series attributes found for {selected_component_type} in the selected networks.")
            return

        selected_attribute = st.sidebar.selectbox(
            f"Select Attribute for {selected_component_type} (e.g., p, p_set, e, marginal_price):",
            options=available_attributes,
            key="ts_attribute_selector"
        )

        if selected_attribute:
            all_component_names_in_type = set()
            for n_obj in networks_to_compare.values():
                comp_info_for_net = get_time_series_component_info(n_obj)
                if selected_component_type in comp_info_for_net:
                    df_attr_prefix = comp_info_for_net[selected_component_type]['df_attr_prefix']
                    if hasattr(getattr(n_obj, df_attr_prefix), selected_attribute):
                        ts_data_for_attr = getattr(getattr(n_obj, df_attr_prefix), selected_attribute)
                        if isinstance(ts_data_for_attr, pd.DataFrame):
                            all_component_names_in_type.update(ts_data_for_attr.columns.tolist())
                        elif isinstance(ts_data_for_attr, pd.Series) and ts_data_for_attr.name:
                            all_component_names_in_type.add(ts_data_for_attr.name)

            if not all_component_names_in_type:
                st.info(f"No individual {selected_component_type}s with '{selected_attribute}' data found across selected scenarios for time-series plotting. Try selecting 'Sum' or 'Total' if available attributes include aggregated data.")
                return

            selected_ts_components = st.sidebar.multiselect(
                f"Select specific {selected_component_type}s:",
                options=sorted(list(all_component_names_in_type)),
                key="ts_individual_component_selector"
            )

            if selected_ts_components:
                combined_ts_df = pd.DataFrame()
                for s_name, n_obj in networks_to_compare.items():
                    ts_data = extract_selected_time_series(n_obj, selected_component_type, selected_attribute, selected_ts_components)
                    if not ts_data.empty:
                        for comp in ts_data.columns:
                            combined_ts_df[f"{comp} ({s_name})"] = ts_data[comp]
                
                if not combined_ts_df.empty:
                    if not isinstance(combined_ts_df.index, pd.DatetimeIndex):
                         combined_ts_df.index = pd.to_datetime(combined_ts_df.index)

                    df_plot = combined_ts_df.reset_index().melt('index', var_name='Component (Scenario)', value_name='Value')
                    df_plot['Component'] = df_plot['Component (Scenario)'].apply(lambda x: x.split(' (')[0].strip())
                    df_plot['Scenario'] = df_plot['Component (Scenario)'].apply(lambda x: x.split(' (')[1][:-1].strip())

                    fig_ts = px.line(df_plot, x='index', y='Value', color='Component', line_dash='Scenario',
                                     title=f'Time Series of {selected_attribute} for {selected_component_type}s',
                                     labels={'index': 'Time', 'Value': selected_attribute})
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("No time series data extracted for the selected components/scenarios.")
            else:
                st.info("Please select specific components for time-series plotting.")
        else:
            st.info("Please select a time-series attribute.")
    else:
        st.info("Please select a component type.")
    
    st.markdown("---")
    
    # --- Predefined Dynamic Plots for Insights ---
    st.subheader("Insightful Dynamic Plots")

    # 1. Stacked Generation Mix over Time
    st.markdown("#### Annual Generation Mix Comparison (Stacked Bar)")
    combined_gen_mix_data = []
    for s_name, n_obj in networks_to_compare.items():
        if not n_obj.generators_t.p.empty:
            df_gen_t_agg = n_obj.generators_t.p.groupby(n_obj.generators.carrier, axis=1).sum()
            df_gen_t_annual_sum = df_gen_t_agg.sum().reset_index()
            df_gen_t_annual_sum.columns = ['Carrier', 'Annual Generation (MWh)']
            df_gen_t_annual_sum['Scenario'] = s_name
            combined_gen_mix_data.append(df_gen_t_annual_sum)

    if combined_gen_mix_data:
        df_combined_gen_mix = pd.concat(combined_gen_mix_data)
        fig_stacked_gen = px.bar(df_combined_gen_mix, x='Scenario', y='Annual Generation (MWh)', color='Carrier',
                                 title='Annual Generation Mix by Scenario',
                                 labels={'Annual Generation (MWh)': 'Annual Generation (MWh)'})
        st.plotly_chart(fig_stacked_gen, use_container_width=True)
    else:
        st.info("No generation data available for stacked generation mix plot.")

    # 2. Load Duration Curve Comparison (Aggregated Load)
    st.markdown("#### Load Duration Curve Comparison")
    combined_ldc_data = []
    for s_name, n_obj in networks_to_compare.items():
        if not n_obj.loads_t.p_set.empty:
            total_load_profile = n_obj.loads_t.p_set.sum(axis=1).sort_values(ascending=False)
            df_ldc = pd.DataFrame({'Load (MW)': total_load_profile.values, 'Duration': range(1, len(total_load_profile) + 1)})
            df_ldc['Scenario'] = s_name
            combined_ldc_data.append(df_ldc)
    
    if combined_ldc_data:
        df_combined_ldc = pd.concat(combined_ldc_data)
        fig_ldc = px.line(df_combined_ldc, x='Duration', y='Load (MW)', color='Scenario',
                          title='Load Duration Curve Comparison',
                          labels={'Load (MW)': 'Load (MW)', 'Duration': 'Duration (hours)'})
        st.plotly_chart(fig_ldc, use_container_width=True)
    else:
        st.info("No load data available for Load Duration Curve comparison.")
    
    # 3. Storage SOC for a specific store over time (Requires user selection for store)
    st.markdown("#### Storage SOC for Selected Stores over Time")
    stores_in_networks = set()
    for n_obj in networks_to_compare.values():
        if not n_obj.stores.empty:
            stores_in_networks.update(n_obj.stores.index.tolist())
    
    if stores_in_networks:
        selected_stores_soc = st.sidebar.multiselect(
            "Select Stores for SOC Time Series:",
            options=sorted(list(stores_in_networks)),
            key="stores_soc_selector"
        )
        if selected_stores_soc:
            combined_soc_df = pd.DataFrame()
            for s_name, n_obj in networks_to_compare.items():
                ts_data = extract_selected_time_series(n_obj, 'Store', 'e', selected_stores_soc)
                if not ts_data.empty:
                    for store in ts_data.columns:
                        combined_soc_df[f"{store} ({s_name})"] = ts_data[store] / 1000 # To GWh
            if not combined_soc_df.empty:
                if not isinstance(combined_soc_df.index, pd.DatetimeIndex):
                     combined_soc_df.index = pd.to_datetime(combined_soc_df.index)

                df_plot_soc = combined_soc_df.reset_index().melt('index', var_name='Store (Scenario)', value_name='SOC (GWh)')
                df_plot_soc['Store'] = df_plot_soc['Store (Scenario)'].apply(lambda x: x.split(' (')[0])
                df_plot_soc['Scenario'] = df_plot_soc['Store (Scenario)'].apply(lambda x: x.split(' (')[1][:-1])

                fig_soc_comp = px.line(df_plot_soc, x='index', y='SOC (GWh)', color='Store', line_dash='Scenario',
                                       title='Selected Storage State of Charge Comparison (GWh)',
                                       labels={'index': 'Time'})
                st.plotly_chart(fig_soc_comp, use_container_width=True)
            else:
                st.info("No SOC data extracted for the selected stores/scenarios.")
        else:
            st.info("Select stores to visualize their SOC over time.")
    else:
        st.info("No storage components found in the loaded networks for SOC comparison.")


    # --- Navigation buttons ---
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("< Back", key="compare_back"):
            st.session_state.current_tab_index = 3
            st.rerun()
    # No "Next" button for the last tab