import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import io
import time
from backend.model_runner import run_model, get_renewable_carriers # Keep get_renewable_carriers here
import os
from datetime import datetime
import shutil
import linopy # For checking available solvers
import plotly.express as px # For plotting results
import plotly.graph_objects as go # For more custom plots

# Helper function to create a Folium map with markers
def create_network_map(df_buses, df_new_generators=None):
    if df_buses.empty:
        return None

    valid_buses = df_buses.dropna(subset=['lat', 'lon'])
    if valid_buses.empty:
        return folium.Map(location=[0, 0], zoom_start=2)
        
    map_center = [valid_buses['lat'].mean(), valid_buses['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=6)

    for idx, row in valid_buses.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"Bus: {row['name']}",
            tooltip=row['name']
        ).add_to(m)
    
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

                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=radius,
                    color='darkorange',
                    fill=True,
                    fill_color='orange',
                    fill_opacity=0.7,
                    popup=f"New Gen: {row['name']}<br>Carrier: {row['carrier']}<br>Capacity: {row['p_nom_opt']:.2f} MW",
                    tooltip=f"New {row['carrier']} ({row['p_nom_opt']:.2f} MW)"
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
                    st.session_state.project_data.get('solver')]):
            st.error("Missing mandatory project or scenario details. Please complete the 'Project' tab.")
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

            data_for_model['df_scenario_year'] = pd.DataFrame({'Scenario': [st.session_state.project_data['scenario_number']], 'Year': [2023]})

            for item in run_model(
                project_name=st.session_state.project_data['project_name'],
                results_dir=st.session_state.project_data['results_dir'],
                solver=st.session_state.project_data['solver'],
                co2_cap=st.session_state.project_data['co2_cap'],
                re_share=st.session_state.project_data['re_share'],
                slack_cost=st.session_state.project_data['slack_cost'],
                discount_rate=st.session_state.project_data['discount_rate'],
                demand_growth=st.session_state.project_data['demand_growth'],
                tech_cost_multipliers=st.session_state.project_data['tech_cost_multipliers'],
                scenario_name=st.session_state.project_data['scenario_name'],
                scenario_number=st.session_state.project_data['scenario_number'],
                line_expansion=st.session_state.project_data['line_expansion'],
                enabled_techs=st.session_state.project_data['enabled_techs'],
                default_new_gen_extendable=st.session_state.project_data['default_new_gen_extendable'],
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
            
            bus_df_for_map = original_df_buses[['Bus name', 'x', 'y']].rename(columns={'x': 'lon', 'y': 'lat', 'Bus name': 'name'})
            bus_df_for_map = bus_df_for_map.dropna(subset=['lat', 'lon'])

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


            map_object = create_network_map(bus_df_for_map, df_new_generators_map)
            if map_object:
                folium_static(map_object)
            else:
                st.info("No valid bus coordinates to display the network map.")

        with tab_plots:
            st.subheader("Simulation Results (Plots)")

            # --- Extract common data needed for multiple plots and key metrics ---
            # These values should be consistently calculated from the network object
            # and potentially stored in session_state or passed from model_runner for consistency
            
            # Recalculate directly from n_results for consistency
            total_annual_demand_MWh = n_results.loads_t.p_set.sum().sum() if not n_results.loads_t.p_set.empty else 0
            
            # Renewable generation in MWh
            renewable_generation_MWh = n_results.generators_t.p.loc[:, n_results.generators.carrier.isin(get_renewable_carriers())].sum().sum() if not n_results.generators_t.p.empty else 0
            
            # Total CO2 emissions in tons/year
            total_co2_emissions_tons = (n_results.generators_t.p.sum().groupby(n_results.generators.carrier).sum() * n_results.carriers.co2_emissions).sum() if not n_results.generators_t.p.empty and 'co2_emissions' in n_results.carriers.columns else 0

            # Installed capacity series (MW)
            installed_capacity_series = n_results.generators.groupby('carrier').p_nom_opt.sum() if not n_results.generators.empty else pd.Series()
            
            # Total generation series (GWh/year)
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

            st.markdown("---") # Separator


            # --- Plot 1: Optimal Generation Capacity & Investment Decisions ---
            st.markdown("### 1. Optimal Generation Capacity & Investment Decisions")
            if not installed_capacity_series.empty:
                df_capacity = installed_capacity_series.reset_index(name='Capacity (MW)')
                fig1 = px.bar(df_capacity, x='carrier', y='Capacity (MW)',
                              title='Optimized Total Installed Capacity by Carrier', labels={'carrier': 'Carrier'})
                st.plotly_chart(fig1, use_container_width=True)

                st.markdown("#### Investment Decisions Table")
                df_investment_decisions = n_results.generators[
                    (n_results.generators.p_nom_extendable == True) &
                    (n_results.generators.p_nom_opt > 0)
                ].copy()
                
                if not df_investment_decisions.empty:
                    df_investment_decisions = df_investment_decisions[[
                        'bus', 'carrier', 'p_nom_initial', 'p_nom_opt', 'capital_cost'
                    ]].rename(columns={
                        'p_nom_initial': 'Initial Capacity (MW)',
                        'p_nom_opt': 'Optimized Capacity (MW)',
                        'capital_cost': 'Annuitized Capital Cost (USD/MW/year)'
                    })
                    df_investment_decisions['New Capacity Built (MW)'] = df_investment_decisions['Optimized Capacity (MW)'] - df_investment_decisions['Initial Capacity (MW)']
                    df_investment_decisions['Annual Investment Cost (USD/year)'] = df_investment_decisions['New Capacity Built (MW)'] * df_investment_decisions['Annuitized Capital Cost (USD/MW/year)']
                    
                    st.dataframe(df_investment_decisions[['bus', 'carrier', 'Initial Capacity (MW)', 'Optimized Capacity (MW)', 'New Capacity Built (MW)', 'Annual Investment Cost (USD/year)']])
                else:
                    st.info("No new generator capacity was built in this scenario.")
            else:
                st.info("No generator capacity data available for plotting investment decisions.")


            # --- Plot 2: Technology Mix (Capacity & Generation Shares) ---
            st.markdown("### 2. Technology Mix (Capacity & Generation Shares)")
            col_cap_mix, col_gen_mix = st.columns(2)
            
            with col_cap_mix:
                st.markdown("#### Capacity Mix (MW)")
                if not installed_capacity_series.empty:
                    df_capacity_mix = installed_capacity_series.reset_index(name='Capacity (MW)')
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
            if n_results.objective is not None and not n_results.generators.empty:
                gen_capital_cost = (n_results.generators.capital_cost * n_results.generators.p_nom_opt).sum()
                
                store_capital_cost = (n_results.stores.capital_cost * n_results.stores.e_nom_opt).sum() if not n_results.stores.empty else 0
                link_capital_cost = (n_results.links.capital_cost * n_results.links.p_nom_opt).sum() if not n_results.links.empty else 0
                line_capital_cost = (n_results.lines.capital_cost * n_results.lines.s_nom_opt).sum() if not n_results.lines.empty else 0
                transformer_capital_cost = (n_results.transformers.capital_cost * n_results.transformers.s_nom).sum() if not n_results.transformers.empty else 0
                total_marginal_cost = (n_results.generators_t.p * n_results.generators.marginal_cost).sum().sum()
                slack_cost_value = (n_results.generators_t.p['slack'] * n_results.generators.loc['slack', 'marginal_cost']).sum() if 'slack' in n_results.generators.index else 0

                calculated_costs = {
                    'Generator Capital (USD/year)': gen_capital_cost,
                    'Generator Marginal (USD/year)': total_marginal_cost,
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
                                  title='Annual System Cost Breakdown', labels={'Amount (USD/year)': 'Amount (USD/year)'})
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No significant cost components found for plotting.")
            else:
                st.info("Cost breakdown data not available (optimization might not have run or failed).")


            # --- Plot 4: CO₂ Emissions (and RE share) ---
            st.markdown("### 4. CO₂ Emissions (and RE share)")
            col_co2, col_re = st.columns(2)

            with col_co2:
                st.markdown("#### Annual CO₂ Emissions")
                co2_cap_value = st.session_state.project_data.get('co2_cap')
                
                data_co2 = {'Metric': ['Total Emissions (tons)', 'CO2 Cap (tons)'],
                            'Value': [total_co2_emissions_tons, co2_cap_value if co2_cap_value is not None else total_co2_emissions_tons * 1.2]}

                df_co2 = pd.DataFrame(data_co2)
                
                fig4_co2 = px.bar(df_co2, x='Metric', y='Value',
                                  title='Annual CO₂ Emissions vs. Cap', labels={'Value': 'Tons CO₂'})
                if co2_cap_value is not None:
                    fig4_co2.add_hline(y=co2_cap_value, line_dash="dash", line_color="red", annotation_text="CO2 Cap", annotation_position="top right")
                st.plotly_chart(fig4_co2, use_container_width=True)
                st.metric(label="Total Annual CO₂ Emissions", value=f"{total_co2_emissions_tons:.2f} tons")

            with col_re:
                st.markdown("#### Renewable Energy Share")
                achieved_re_share = 0
                if total_annual_demand_MWh > 0:
                    achieved_re_share = (renewable_generation_MWh / total_annual_demand_MWh) * 100
                
                re_share_target = st.session_state.project_data.get('re_share')
                re_share_target_percent = re_share_target * 100 if re_share_target is not None else achieved_re_share * 1.1

                data_re = {'Metric': ['Achieved RE Share (%)', 'RE Share Target (%)'],
                           'Value': [achieved_re_share, re_share_target_percent]}
                df_re = pd.DataFrame(data_re)

                fig4_re = px.bar(df_re, x='Metric', y='Value',
                                 title='Achieved RE Share vs. Target', labels={'Value': 'Percentage (%)'})
                if re_share_target is not None:
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
                df_links = n_results.links[n_results.links.carrier.isin(['storage_charge', 'storage_discharge'])].copy()
                if not df_links.empty and not n_results.links_t.p.empty:
                    total_charge_power_t = n_results.links_t.p[df_links[df_links.carrier == 'storage_charge'].index].sum(axis=1)
                    total_discharge_power_t = n_results.links_t.p[df_links[df_links.carrier == 'storage_discharge'].index].sum(axis=1)

                    df_charge_discharge = pd.DataFrame({
                        'Charging Power (MW)': total_charge_power_t.apply(lambda x: -x if x < 0 else 0),
                        'Discharging Power (MW)': total_discharge_power_t.apply(lambda x: x if x > 0 else 0)
                    }, index=n_results.snapshots)
                    
                    df_charge_discharge['Charging Power (MW)'] = df_charge_discharge['Charging Power (MW)'].abs()
                    df_charge_discharge['Discharging Power (MW)'] = df_charge_discharge['Discharging Power (MW)'].abs()

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
                # Aggregate dispatch by carrier
                df_hourly_dispatch = n_results.generators_t.p.groupby(n_results.generators.carrier, axis=1).sum()
                df_hourly_dispatch.index.name = 'Time'

                if not df_hourly_dispatch.empty:
                    # Melt for plotly express stacked area chart
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
            # --- END NEW ---

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