import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import io
import time
from backend.model_runner import run_model, get_renewable_carriers 
import os
from datetime import datetime
import shutil
import linopy # For checking available solvers
import plotly.express as px # For plotting results
import plotly.graph_objects as go # For more custom plots

# Helper function to create a Folium map with markers and connections
def create_network_map(n_results, df_buses, df_new_generators=None): # Now accepts n_results directly for connections
    if df_buses.empty:
        return None

    valid_buses = df_buses.dropna(subset=['lat', 'lon'])
    if valid_buses.empty:
        return folium.Map(location=[0, 0], zoom_start=2)
        
    map_center = [valid_buses['lat'].mean(), valid_buses['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=5, height=900) # CORRECTED: Increased map height to 900 for ~3/4 screen

    # Dictionary to quickly look up bus coordinates by name
    bus_coords = {row['name']: (row['lat'], row['lon']) for idx, row in valid_buses.iterrows()}

    # Add existing bus locations (with enhanced popups)
    for idx, row in valid_buses.iterrows():
        bus_name = row['name']
        popup_html = f"<b>Bus:</b> {bus_name}<br>" \
                     f"<b>V_nom:</b> {row.get('v_nom', 'N/A')}<br>" \
                     f"<b>Carrier:</b> {row.get('carrier', 'N/A')}<br>" \
                     f"<b>Unit:</b> {row.get('unit', 'N/A')}"
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=bus_name
        ).add_to(m)
    
    # Add newly built/expanded generators (with enhanced popups)
    if df_new_generators is not None and not df_new_generators.empty:
        df_new_generators_filtered = df_new_generators.dropna(subset=['lat', 'lon'])
        df_new_generators_filtered = df_new_generators_filtered[df_new_generators_filtered['p_nom_opt'] > 0]
        
        if not df_new_generators_filtered.empty:
            max_capacity = df_new_generators_filtered['p_nom_opt'].max()
            min_capacity = df_new_generators_filtered['p_nom_opt'].min()
            
            for idx, row in df_new_generators_filtered.iterrows():
                if max_capacity > min_capacity and max_capacity != 0:
                    radius = 5 + 15 * ((row['p_nom_opt'] - min_capacity) / (max_capacity - min_capacity))
                else:
                    radius = 10 if max_capacity > 0 else 5
                
                popup_html = f"<b>New Gen:</b> {row['name']}<br>" \
                             f"<b>Bus:</b> {row['bus']}<br>" \
                             f"<b>Carrier:</b> {row['carrier']}<br>" \
                             f"<b>Initial Cap:</b> {row.get('p_nom_initial', 'N/A'):.2f} MW<br>" \
                             f"<b>Optimized Cap:</b> {row['p_nom_opt']:.2f} MW"

                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=radius,
                    color='darkorange',
                    fill=True,
                    fill_color='orange',
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"New {row['carrier']} ({row['p_nom_opt']:.2f} MW)"
                ).add_to(m)

    # --- ADDED: Draw Lines (Transmission Lines) ---
    if hasattr(n_results, 'lines') and not n_results.lines.empty:
        max_s_nom_line = n_results.lines.s_nom_opt.max() if not n_results.lines.s_nom_opt.empty else n_results.lines.s_nom.max() if not n_results.lines.s_nom.empty else 1
        for idx, line in n_results.lines.iterrows():
            bus0_name = line['bus0']
            bus1_name = line['bus1']
            if bus0_name in bus_coords and bus1_name in bus_coords:
                points = [bus_coords[bus0_name], bus_coords[bus1_name]]
                popup_html = f"<b>Line:</b> {idx}<br>" \
                             f"<b>From:</b> {bus0_name}<br>" \
                             f"<b>To:</b> {bus1_name}<br>" \
                             f"<b>Type:</b> {line.get('type', 'N/A')}<br>" \
                             f"<b>S_nom:</b> {line.get('s_nom', 'N/A'):.2f} MVA<br>" \
                             f"<b>S_nom_opt:</b> {line.get('s_nom_opt', line.get('s_nom', 'N/A')):.2f} MVA<br>" \
                             f"<b>Length:</b> {line.get('length', 'N/A'):.2f} km"
                
                line_weight = line.get('s_nom_opt', line.get('s_nom', 1)) / max_s_nom_line * 5 # Scale weight, max 5px
                folium.PolyLine(
                    locations=points,
                    color='blue',
                    weight=max(1, line_weight), # Minimum weight 1 for visibility
                    opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Line {idx}: {bus0_name}-{bus1_name} ({line.get('s_nom_opt', line.get('s_nom', 'N/A')):.0f} MVA)"
                ).add_to(m)

    # --- ADDED: Draw Links (e.g., Storage Links) ---
    if hasattr(n_results, 'links') and not n_results.links.empty:
        max_p_nom_link = n_results.links.p_nom_opt.max() if not n_results.links.p_nom_opt.empty else n_results.links.p_nom.max() if not n_results.links.p_nom.empty else 1
        for idx, link in n_results.links.iterrows():
            bus0_name = link['bus0']
            bus1_name = link['bus1']
            if bus0_name in bus_coords and bus1_name in bus_coords:
                points = [bus_coords[bus0_name], bus_coords[bus1_name]]
                popup_html = f"<b>Link:</b> {idx}<br>" \
                             f"<b>From:</b> {bus0_name}<br>" \
                             f"<b>To:</b> {bus1_name}<br>" \
                             f"<b>Carrier:</b> {link.get('carrier', 'N/A')}<br>" \
                             f"<b>P_nom:</b> {link.get('p_nom', 'N/A'):.2f} MW<br>" \
                             f"<b>P_nom_opt:</b> {link.get('p_nom_opt', 'N/A'):.2f} MW"
                
                link_color = 'green' if link.get('carrier') == 'battery_link' else 'purple'
                link_weight = link.get('p_nom_opt', link.get('p_nom', 1)) / max_p_nom_link * 4 # Scale weight, max 4px

                folium.PolyLine(
                    locations=points,
                    color=link_color,
                    weight=max(1, link_weight),
                    opacity=0.7,
                    dash_array='5, 5', # Dashed lines for links
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Link {idx}: {bus0_name}-{bus1_name} ({link.get('p_nom_opt', link.get('p_nom', 'N/A')):.0f} MW)"
                ).add_to(m)
    
    # --- ADDED: Draw Transformers ---
    if hasattr(n_results, 'transformers') and not n_results.transformers.empty:
        max_s_nom_transformer = n_results.transformers.s_nom.max() if not n_results.transformers.s_nom.empty else 1
        for idx, transformer in n_results.transformers.iterrows():
            bus0_name = transformer['bus0']
            bus1_name = transformer['bus1']
            if bus0_name in bus_coords and bus1_name in bus_coords:
                points = [bus_coords[bus0_name], bus_coords[bus1_name]]
                popup_html = f"<b>Transformer:</b> {idx}<br>" \
                             f"<b>From:</b> {bus0_name}<br>" \
                             f"<b>To:</b> {bus1_name}<br>" \
                             f"<b>S_nom:</b> {transformer.get('s_nom', 'N/A'):.2f} MVA"
                
                transformer_weight = transformer.get('s_nom', 1) / max_s_nom_transformer * 3 # Scale weight, max 3px
                folium.PolyLine(
                    locations=points,
                    color='gray',
                    weight=max(1, transformer_weight),
                    opacity=0.8,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Transformer {idx}: {bus0_name}-{bus1_name} ({transformer.get('s_nom', 'N/A') :.0f} MVA)"
                ).add_to(m)

    return m

def show_tab():
    st.title("Run Simulation & View Outputs")

    if st.session_state.highs_path_set_status:
        st.info("HiGHS solver path successfully configured at application startup.")
    else:
        st.warning("HiGHS solver path might not be automatically configured. Ensure 'highs' is in system PATH if using HiGHS.")
    
    try:
        if hasattr(linopy, 'available_solvers'):
            st.info(f"Available solvers detected by Linopy: {', '.join(linopy.available_solvers)}")
        else:
             st.warning("Could not determine available solvers from Linopy.")
    except Exception as e:
        st.warning(f"Error checking Linopy available solvers: {e}")


    st.subheader("Simulation Control")

    if st.button("Run Simulation", key="run_simulation_button", help="Click to start the PyPSA optimization."):
        if not all([st.session_state.project_data.get('project_name'),
                    st.session_state.project_data.get('results_dir'),
                    st.session_state.project_data.get('scenario_name'),
                    st.session_state.project_data.get('solver'),
                    st.session_state.project_data.get('scenario_year') is not None,
                    (st.session_state.project_data.get('demand_projection_method') == "Target Peak Demand" and st.session_state.project_data.get('target_peak_demand') is not None) or \
                    (st.session_state.project_data.get('demand_projection_method') == "Percentage Growth" and st.session_state.project_data.get('demand_growth_percentage') is not None)
                    ]):
            st.error("Missing mandatory project or scenario details (Project Name, Results Dir, Scenario Name, Solver, Scenario Year, Demand Projection). Please complete the 'Project' tab.")
            return

        has_bus_data = False
        if st.session_state.data_mapping_mode.get('buses') == "Excel Mapping":
            if st.session_state.mapped_data.get('buses', {}).get('df_content'): has_bus_data = True
        elif st.session_state.data_mapping_mode.get('buses') == "Manual Entry":
            if not st.session_state.manual_data.get('buses', pd.DataFrame()).empty: has_bus_data = True

        has_demand_data = False
        if st.session_state.data_mapping_mode.get('demand') == "Excel Mapping":
            if st.session_state.mapped_data.get('demand', {}).get('df_content'): has_demand_data = True
        elif st.session_state.data_mapping_mode.get('demand') == "Manual Entry":
            if not st.session_state.manual_data.get('demand', pd.DataFrame()).empty: has_demand_data = True
        
        if not has_bus_data:
            st.error("No bus data found. Please provide bus data in the 'Data Mapping' tab.")
            return
        if not has_demand_data:
            st.error("No demand data found. Please provide demand data in the 'Data Mapping' tab.")
            return
        
        st.info("Validation passed. Starting simulation...")
        
        st.session_state.log_output = ""
        log_placeholder = st.empty()

        final_n_results = None
        final_results_prefix = None

        try:
            data_for_model = {}
            component_to_df_map = {
                "buses": "df_buses",
                "generators": "df_generators",
                "demand": "df_load",
                "transmission_lines": "df_transmission_lines",
                "transformers": "df_transformers",
                "storage": "df_storage",
                "generation_profiles": "df_generation_profiles"
            }

            for comp_type, df_key in component_to_df_map.items():
                if comp_type == "demand" or comp_type == "generation_profiles":
                    if st.session_state.data_mapping_mode.get(comp_type) == "Excel Mapping":
                        mapped = st.session_state.mapped_data.get(comp_type, {})
                        data_for_model[df_key] = pd.DataFrame(mapped['df_content']) if mapped.get('df_content') else pd.DataFrame()
                        if mapped.get('sheet_name'):
                            data_for_model[f'{df_key}_mapping'] = mapped
                    else:
                        data_for_model[df_key] = st.session_state.manual_data.get(comp_type, pd.DataFrame())
                        data_for_model[f'{df_key}_mapping'] = {}
                else:
                    if st.session_state.data_mapping_mode.get(comp_type) == "Excel Mapping":
                        mapped = st.session_state.mapped_data.get(comp_type, {})
                        if mapped.get('df_content'):
                            df_raw = pd.DataFrame(mapped['df_content'])
                            rename_dict = {
                                mapped_col: default_col
                                for default_col, mapped_col in mapped.items()
                                if default_col != 'sheet_name' and default_col != 'df_content' and mapped_col in df_raw.columns
                            }
                            data_for_model[df_key] = df_raw.rename(columns=rename_dict).copy()
                        else:
                            data_for_model[df_key] = pd.DataFrame()
                    else:
                        data_for_model[df_key] = st.session_state.manual_data.get(comp_type, pd.DataFrame()).copy()

            data_for_model['df_scenario_year'] = pd.DataFrame({'Scenario': [st.session_state.project_data['scenario_number']], 'Year': [st.session_state.project_data['scenario_year']]}) 

            for item in run_model(
                data_file=io.BytesIO(st.session_state.excel_file_buffer),
                results_dir=st.session_state.project_data['results_dir'],
                solver=st.session_state.project_data['solver'],
                co2_cap=st.session_state.project_data['co2_cap'],
                re_share=st.session_state.project_data['re_share'],
                slack_cost=st.session_state.project_data['slack_cost'],
                discount_rate=st.session_state.project_data['discount_rate'],
                tech_cost_multipliers=st.session_state.project_data['tech_cost_multipliers'],
                scenario_name=st.session_state.project_data['scenario_name'],
                scenario_number=st.session_state.project_data['scenario_number'],
                line_expansion=st.session_state.project_data['line_expansion'],
                enabled_techs=st.session_state.project_data['enabled_techs'],
                default_new_gen_extendable=st.session_state.project_data['default_new_gen_extendable'],
                scenario_year=st.session_state.project_data['scenario_year'],
                target_peak_demand=st.session_state.project_data['target_peak_demand'],
                demand_projection_method=st.session_state.project_data['demand_projection_method'],
                demand_growth_percentage=st.session_state.project_data['demand_growth_percentage'],
                df_buses=data_for_model.get('df_buses', pd.DataFrame()),
                df_generators=data_for_model.get('df_generators', pd.DataFrame()),
                df_load=data_for_model.get('df_load', pd.DataFrame()),
                df_transmission_lines=data_for_model.get('df_transmission_lines', pd.DataFrame()),
                df_transformers=data_for_model.get('df_transformers', pd.DataFrame()),
                df_storage=data_for_model.get('df_storage', pd.DataFrame()),
                df_generation_profiles=data_for_model.get('df_generation_profiles', pd.DataFrame()),
                df_scenario_year=data_for_model.get('df_scenario_year', pd.DataFrame())
            ):
                if isinstance(item, tuple) and len(item) == 2:
                    final_n_results, final_results_prefix = item
                    st.session_state.log_output += f"[{datetime.now().strftime('%H:%M:%S')}] Final results object received from model.\n"
                else:
                    st.session_state.log_output += str(item) + "\n"
                log_placeholder.code(st.session_state.log_output, language="text")
                time.sleep(0.01)

            if final_n_results is not None and final_results_prefix is not None:
                st.success("Simulation completed successfully! Results are saved to disk.")
                st.session_state.simulation_results = {
                    'network_object': final_n_results,
                    'results_path_prefix': final_results_prefix,
                    'initial_bus_data': data_for_model.get('df_buses', pd.DataFrame())
                }
            else:
                st.error("Simulation failed: Could not retrieve final results object from the model.")
                st.session_state.simulation_results = None
            
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.exception(e)
            st.session_state.log_output += f"\nERROR: {e}\n"
            log_placeholder.code(st.session_state.log_output, language="text") 

    st.subheader("Live Simulation Log")

    if st.session_state.simulation_results and st.session_state.simulation_results.get('network_object'):
        n_results = st.session_state.simulation_results['network_object']
        original_df_buses = st.session_state.simulation_results['initial_bus_data']
        results_path_prefix = st.session_state.simulation_results['results_path_prefix']

        tab_map, tab_plots = st.tabs(["Network Overview (Map)", "Simulation Results (Plots)"])

        with tab_map:
            st.subheader("Network Overview (Map)")
            
            bus_df_for_map = original_df_buses.copy()
            bus_df_for_map = bus_df_for_map.rename(columns={'Bus name': 'name', 'x': 'lon', 'y': 'lat'})
            bus_df_for_map = bus_df_for_map.dropna(subset=['lon', 'lat'])

            df_optimized_generators = n_results.generators[n_results.generators.p_nom_opt > 0].copy()
            df_new_generators_map = pd.DataFrame()

            if not df_optimized_generators.empty:
                df_new_generators_map = df_optimized_generators.merge(
                    n_results.buses[['x', 'y']], left_on='bus', right_index=True, how='left'
                ).rename(columns={'x': 'lon', 'y': 'lat'})
                df_new_generators_map['name'] = df_new_generators_map.index

                df_new_generators_map = df_new_generators_map[
                    (df_new_generators_map['p_nom_extendable'] == True) &
                    (df_new_generators_map['p_nom_opt'] > 0)
                ].copy()

            # Pass n_results to the map function for connections
            map_object = create_network_map(n_results, bus_df_for_map, df_new_generators_map)
            if map_object:
                folium_static(map_object)
            else:
                st.info("No valid bus coordinates to display the network map.")

        with tab_plots:
            st.subheader("Simulation Results (Plots)")

            total_annual_demand_MWh = n_results.loads_t.p_set.sum().sum() if not n_results.loads_t.p_set.empty else 0
            renewable_generation_MWh = n_results.generators_t.p.loc[:, n_results.generators.carrier.isin(get_renewable_carriers())].sum().sum() if not n_results.generators_t.p.empty else 0
            total_co2_emissions_tons = (n_results.generators_t.p.sum().groupby(n_results.generators.carrier).sum() * n_results.carriers.co2_emissions).sum() if not n_results.generators_t.p.empty and 'co2_emissions' in n_results.carriers.columns else 0

            installed_capacity_series = n_results.generators.groupby('carrier').p_nom_opt.sum() if not n_results.generators.empty else pd.Series()
            total_generation_series_GWh = n_results.generators_t.p.sum().groupby(n_results.generators.carrier).sum() / 1e3 if not n_results.generators_t.p.empty else pd.Series()
            
            # --- Key Metrics Summary Table ---
            st.markdown("### Key Metrics Summary")
            metrics_data = {
                "Metric": [],
                "Value": [],
                "Unit": []
            }
            
            if n_results.objective is not None:
                metrics_data["Metric"].append("Total System Cost")
                metrics_data["Value"].append(f"{n_results.objective:.2f}")
                metrics_data["Unit"].append("USD")
            
            if not installed_capacity_series.empty:
                metrics_data["Metric"].append("Total Installed Generation Capacity")
                metrics_data["Value"].append(f"{installed_capacity_series.sum():.2f}")
                metrics_data["Unit"].append("MW")

            if not total_generation_series_GWh.empty:
                metrics_data["Metric"].append("Total Annual Generation")
                metrics_data["Value"].append(f"{total_generation_series_GWh.sum():.2f}")
                metrics_data["Unit"].append("GWh/year")
            
            if total_annual_demand_MWh > 0:
                achieved_re_share = (renewable_generation_MWh / total_annual_demand_MWh) * 100
                metrics_data["Metric"].append("Achieved RE Share")
                metrics_data["Value"].append(f"{achieved_re_share:.2f}")
                metrics_data["Unit"].append("%")
            
            metrics_data["Metric"].append("Total Annual CO₂ Emissions")
            metrics_data["Value"].append(f"{total_co2_emissions_tons:.2f}")
            metrics_data["Unit"].append("tons/year")

            df_key_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_key_metrics, hide_index=True, use_container_width=True)

            st.markdown("---")


            # --- Plot 1: Optimal Generation Capacity & Investment Decisions ---
            st.markdown("### 1. Optimal Generation Capacity & Investment Decisions")
            if not installed_capacity_series.empty:
                df_capacity = installed_capacity_series.reset_index(name='Capacity (MW)')
                if 'slack' in df_capacity['carrier'].values: # CORRECTED: Drop slack from capacity plot
                    df_capacity = df_capacity[df_capacity['carrier'] != 'slack']

                fig1 = px.bar(df_capacity, x='carrier', y='Capacity (MW)',
                              title='Optimized Total Installed Capacity by Carrier', labels={'carrier': 'Carrier'})
                st.plotly_chart(fig1, use_container_width=True)

                st.markdown("#### Investment Decisions Table")
                df_investment_decisions = n_results.generators[
                    (n_results.generators.p_nom_extendable == True) &
                    (n_results.generators.p_nom_opt > 0) # Only consider investments where opt capacity > 0
                ].copy()
                
                if not df_investment_decisions.empty:
                    df_investment_decisions['New Capacity Built (MW)'] = df_investment_decisions['p_nom_opt'] - df_investment_decisions['p_nom_initial']
                    # CORRECTED: Filter for positive new builds and calculate investment cost ONLY for positive builds
                    df_investment_decisions = df_investment_decisions[df_investment_decisions['New Capacity Built (MW)'] > 0].copy()

                    if not df_investment_decisions.empty:
                        df_investment_decisions['Annual Investment Cost (USD/year)'] = df_investment_decisions['New Capacity Built (MW)'] * df_investment_decisions['capital_cost'] # capital_cost is already annuitized

                        df_investment_decisions = df_investment_decisions[[
                            'bus', 'carrier', 'p_nom_initial', 'p_nom_opt', 'New Capacity Built (MW)', 'Annual Investment Cost (USD/year)'
                        ]].rename(columns={
                            'p_nom_initial': 'Initial Capacity (MW)',
                            'p_nom_opt': 'Optimized Capacity (MW)',
                        })
                        st.dataframe(df_investment_decisions)
                    else:
                        st.info("No new generator capacity was built (or expanded) in this scenario.")
                else:
                    st.info("No new generator capacity was built (or expanded) in this scenario.")
            else:
                st.info("No generator capacity data available for plotting investment decisions.")


            # --- Plot 2: Technology Mix (Capacity & Generation Shares) ---
            st.markdown("### 2. Technology Mix (Capacity & Generation Shares)")
            col_cap_mix, col_gen_mix = st.columns(2)
            
            with col_cap_mix:
                st.markdown("#### Capacity Mix (MW)")
                if not installed_capacity_series.empty:
                    df_capacity_mix = installed_capacity_series.reset_index(name='Capacity (MW)')
                    if 'slack' in df_capacity_mix['carrier'].values: # Drop slack from pie chart
                        df_capacity_mix = df_capacity_mix[df_capacity_mix['carrier'] != 'slack']
                    
                    fig2_cap = px.pie(df_capacity_mix, values='Capacity (MW)', names='carrier',
                                      title='Optimized Capacity Mix', hole=0.3)
                    st.plotly_chart(fig2_cap, use_container_width=True)
                else:
                    st.info("No capacity mix data available.")

            with col_gen_mix:
                st.markdown("#### Annual Generation Share (GWh/year)")
                if not total_generation_series_GWh.empty:
                    df_generation_mix = total_generation_series_GWh.reset_index(name='Generation (GWh/year)')
                    fig2_gen = px.pie(df_generation_mix, values='Generation (GWh/year)', names='carrier',
                                      title='Annual Generation Share', hole=0.3)
                    st.plotly_chart(fig2_gen, use_container_width=True)
                else:
                    st.info("No generation mix data available.")


            # --- Plot 3: Cost Breakdown ---
            st.markdown("### 3. Cost Breakdown")
            cost_breakdown_view = st.radio(
                "Cost Breakdown View:",
                ("By Cost Type", "By Carrier (Generators)"),
                key="cost_breakdown_view_toggle"
            )

            if cost_breakdown_view == "By Cost Type":
                if n_results.objective is not None and not n_results.generators.empty:
                    gen_capital_cost = (n_results.generators.capital_cost * n_results.generators.p_nom_opt).sum()
                    gen_fixed_operation_cost = (n_results.generators.fixed_operation_cost * n_results.generators.p_nom_opt).sum() if 'fixed_operation_cost' in n_results.generators.columns else 0
                    
                    store_capital_cost = (n_results.stores.capital_cost * n_results.stores.e_nom_opt).sum() if not n_results.stores.empty else 0
                    link_capital_cost = (n_results.links.capital_cost * n_results.links.p_nom_opt).sum() if not n_results.links.empty else 0
                    line_capital_cost = (n_results.lines.capital_cost * n_results.lines.s_nom_opt).sum() if not n_results.lines.empty else 0
                    transformer_capital_cost = (n_results.transformers.capital_cost * n_results.transformers.s_nom).sum() if not n_results.transformers.empty else 0
                    total_marginal_cost = (n_results.generators_t.p * n_results.generators.marginal_cost).sum().sum()
                    slack_cost_value = (n_results.generators_t.p['slack'] * n_results.generators.loc['slack', 'marginal_cost']).sum() if 'slack' in n_results.generators.index else 0

                    calculated_costs = {
                        'Generator Capital (USD/year)': gen_capital_cost,
                        'Generator Fixed O&M (USD/year)': gen_fixed_operation_cost,
                        'Generator Variable (USD/year)': total_marginal_cost,
                        'Storage Capital (USD/year)': store_capital_cost,
                        'Link Capital (USD/year)': link_capital_cost,
                        'Line Capital (USD/year)': line_capital_cost,
                        'Transformer Capital (USD/year)': transformer_capital_cost,
                        'Slack Cost (USD/year)': slack_cost_value
                    }
                    
                    df_costs_breakdown = pd.DataFrame(list(calculated_costs.items()), columns=['Cost Type', 'Amount (USD/year)'])
                    df_costs_breakdown = df_costs_breakdown[df_costs_breakdown['Amount (USD/year)'] > 0]

                    if not df_costs_breakdown.empty:
                        fig3 = px.bar(df_costs_breakdown, x='Cost Type', y='Amount (USD/year)',
                                      title='Annual System Cost Breakdown - By Cost Type', labels={'Amount (USD/year)': 'Amount (USD/year)'})
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.info("No significant cost components found for plotting.")
                else:
                    st.info("Cost breakdown data not available (optimization might not have run or failed).")
            
            else: # By Carrier (Generators)
                st.markdown("#### Annual System Cost Breakdown - By Generator Carrier")
                if not n_results.generators.empty and not n_results.generators_t.p.empty:
                    df_gen_costs = n_results.generators.copy()
                    df_gen_costs['annual_capital_cost'] = df_gen_costs['capital_cost'] * df_gen_costs['p_nom_opt']
                    df_gen_costs['annual_fixed_om_cost'] = df_gen_costs['fixed_operation_cost'] * df_gen_costs['p_nom_opt'] if 'fixed_operation_cost' in df_gen_costs.columns else 0
                    
                    gen_annual_dispatch_MWh = n_results.generators_t.p.sum()
                    df_gen_costs['annual_variable_cost'] = gen_annual_dispatch_MWh * df_gen_costs['marginal_cost']
                    df_gen_costs['total_annual_cost'] = df_gen_costs['annual_capital_cost'] + df_gen_costs['annual_fixed_om_cost'] + df_gen_costs['annual_variable_cost']

                    df_costs_by_carrier = df_gen_costs.groupby('carrier')['total_annual_cost'].sum().reset_index(name='Total Annual Cost (USD/year)')
                    
                    if 'slack' in df_costs_by_carrier['carrier'].values:
                        df_costs_by_carrier = df_costs_by_carrier[df_costs_by_carrier['carrier'] != 'slack']

                    if not df_costs_by_carrier.empty:
                        fig3_carrier = px.bar(df_costs_by_carrier, x='carrier', y='Total Annual Cost (USD/year)',
                                              title='Annual System Cost by Generator Carrier', labels={'carrier': 'Carrier'})
                        st.plotly_chart(fig3_carrier, use_container_width=True)
                    else:
                        st.info("No generator carrier costs found for plotting.")
                else:
                    st.info("Generator data not available for cost breakdown by carrier.")


            # --- Plot 4: CO₂ Emissions (and RE share) ---
            st.markdown("### 4. CO₂ Emissions (and RE share)")
            col_co2, col_re = st.columns(2)

            with col_co2:
                st.markdown("#### Annual CO₂ Emissions")
                co2_cap_value = st.session_state.project_data.get('co2_cap')
                
                data_co2_metrics = []
                data_co2_metrics.append({'Metric': 'Total Emissions (tons)', 'Value': total_co2_emissions_tons})
                if co2_cap_value is not None and co2_cap_value > 0: # Corrected: Only add cap if it's active
                    data_co2_metrics.append({'Metric': 'CO2 Cap (tons)', 'Value': co2_cap_value})
                
                df_co2 = pd.DataFrame(data_co2_metrics)
                
                fig4_co2 = px.bar(df_co2, x='Metric', y='Value',
                                  title='Annual CO₂ Emissions vs. Cap', labels={'Value': 'Tons CO₂'})
                if co2_cap_value is not None and co2_cap_value > 0: # Corrected: Only add hline if cap is active
                    fig4_co2.add_hline(y=co2_cap_value, line_dash="dash", line_color="red", annotation_text="CO2 Cap", annotation_position="top right")
                st.plotly_chart(fig4_co2, use_container_width=True)
                st.metric(label="Total Annual CO₂ Emissions", value=f"{total_co2_emissions_tons:.2f} tons")

            with col_re:
                st.markdown("#### Renewable Energy Share")
                achieved_re_share = 0
                if total_annual_demand_MWh > 0:
                    achieved_re_share = (renewable_generation_MWh / total_annual_demand_MWh) * 100
                
                re_share_target = st.session_state.project_data.get('re_share')
                re_share_target_percent = re_share_target * 100 if re_share_target is not None else achieved_re_share * 1.1 # Default if no target set

                data_re_metrics = []
                data_re_metrics.append({'Metric': 'Achieved RE Share (%)', 'Value': achieved_re_share})
                if re_share_target is not None and re_share_target > 0: # Corrected: Only add target if active
                    data_re_metrics.append({'Metric': 'RE Share Target (%)', 'Value': re_share_target * 100})

                df_re = pd.DataFrame(data_re_metrics)

                fig4_re = px.bar(df_re, x='Metric', y='Value',
                                 title='Achieved RE Share vs. Target', labels={'Value': 'Percentage (%)'})
                if re_share_target is not None and re_share_target > 0: # Corrected: Only add hline if target is active
                    fig4_re.add_hline(y=re_share_target*100, line_dash="dash", line_color="green", annotation_text="RE Target", annotation_position="top right")
                st.plotly_chart(fig4_re, use_container_width=True)
                st.metric(label="Achieved RE Share", value=f"{achieved_re_share:.2f}%")


            # --- Plot 5: Storage Behaviour ---
            st.markdown("### 5. Storage Behaviour")
            if not n_results.stores.empty:
                st.markdown("#### Total Storage State of Charge (SOC) over time")
                total_soc_t = n_results.stores_t.e.sum(axis=1) / 1000 # Convert to GWh for better scale
                fig5_soc = px.line(total_soc_t, title='Total System Storage State of Charge (GWh)',
                                   labels={'value': 'Total SOC (GWh)', 'index': 'Time'})
                st.plotly_chart(fig5_soc, use_container_width=True)

                st.markdown("#### Total Storage Charging/Discharging Power over time")
                df_battery_links = n_results.links[n_results.links.carrier == 'battery_link'].copy()
                if not df_battery_links.empty and hasattr(n_results.links_t, 'p0') and not n_results.links_t.p0.empty:
                    total_battery_power = n_results.links_t.p0[df_battery_links.index].sum(axis=1)

                    df_charge_discharge = pd.DataFrame({
                        'Charging Power (MW)': total_battery_power.apply(lambda x: -x if x < 0 else 0),
                        'Discharging Power (MW)': total_battery_power.apply(lambda x: x if x > 0 else 0)
                    }, index=n_results.snapshots)
                    
                    fig5_power = px.area(df_charge_discharge, title='Total System Storage Charging/Discharging Power (MW)',
                                         labels={'value': 'Power (MW)', 'index': 'Time'},
                                         color_discrete_map={'Charging Power (MW)': 'blue', 'Discharging Power (MW)': 'green'})
                    st.plotly_chart(fig5_power, use_container_width=True)
                else:
                    st.info("No storage charging/discharging power data available.")
            else:
                st.info("No storage data available for plotting.")

            # --- NEW: Hourly Dispatch Plot ---
            st.markdown("### 6. Hourly Generation Dispatch")
            if not n_results.generators_t.p.empty:
                df_hourly_dispatch = n_results.generators_t.p.groupby(n_results.generators.carrier, axis=1).sum()
                if 'slack' in df_hourly_dispatch.columns:
                    df_hourly_dispatch = df_hourly_dispatch.drop(columns=['slack'])

                df_hourly_dispatch.index.name = 'Time'

                if not df_hourly_dispatch.empty:
                    df_plot_dispatch = df_hourly_dispatch.reset_index().melt(
                        'Time', var_name='Carrier', value_name='Dispatch (MW)'
                    )
                    fig6_dispatch = px.area(df_plot_dispatch, x='Time', y='Dispatch (MW)', color='Carrier',
                                            title='Hourly Generation Dispatch by Carrier (MW)',
                                            labels={'Dispatch (MW)': 'Dispatch (MW)'},
                                            hover_data={'Dispatch (MW)': ':.2f'})
                    st.plotly_chart(fig6_dispatch, use_container_width=True)
                else:
                    st.info("No hourly dispatch data available for plotting.")
            else:
                st.info("No generator dispatch data available for hourly plotting.")

    else:
        st.info("Run a simulation first to view results.")

    st.subheader("Download All Results")
    if st.session_state.simulation_results and st.session_state.simulation_results.get('results_path_prefix'):
        results_path_prefix = st.session_state.simulation_results['results_path_prefix']
        
        zip_base_name = os.path.basename(results_path_prefix)
        zip_archive_name = f"{results_path_prefix}_all_results"
        
        try:
            shutil.make_archive(zip_archive_name, 'zip', results_path_prefix)
            final_zip_file = f"{zip_archive_name}.zip"

            if os.path.exists(final_zip_file):
                with open(final_zip_file, "rb") as f:
                    st.download_button(
                        label="Download All Results (ZIP)",
                        data=f.read(),
                        file_name=f"{zip_base_name}_all_results.zip",
                        mime="application/zip"
                    )
                st.success("ZIP archive created and ready for download.")
            else:
                st.error(f"Failed to create ZIP archive at {final_zip_file}.")
        except Exception as e:
            st.error(f"Error creating ZIP archive: {e}")
            st.exception(e)

    else:
        st.info("Run a simulation first to enable result downloads.")

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Next >", key="simulation_next"):
            st.session_state.current_tab_index = 4
            st.rerun()
    with col1:
        if st.button("< Back", key="simulation_back"):
            st.session_state.current_tab_index = 2
            st.rerun()