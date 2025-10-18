import pandas as pd
import pypsa
from datetime import datetime
import numpy as np
import os
import ast
import sys
import shutil
import re       # New: For robust filename handling
import calendar # New: For leap year handling
import plotly.graph_objects as go # For plotly HTML plots
import plotly.express as px       # For plotly HTML plots

# --------------------------
# Helper functions
# --------------------------
def calculate_annuity(capital_cost, interest_rate, lifetime):
    """Convert CAPEX to annuitized annualized cost."""
    if lifetime <= 0: # Robustness check
        return capital_cost
    if interest_rate == 0:
        return capital_cost / lifetime
    return (capital_cost * interest_rate) / (1 - (1 + interest_rate) ** -lifetime)

def apply_cost_multiplier(carrier, base_capex, multipliers):
    """
    Scale the base CAPEX (from Excel) with tech-specific multipliers.
    Fallback to 1.0 if carrier not in multipliers.
    """
    key = carrier.lower()
    factor = multipliers.get(key, 1.0)
    return base_capex * factor

def get_renewable_carriers():
    """Define which carriers are considered renewable energy sources"""
    return ["Solar", "Solar Rooftop", "Wind", "Geothermal", "CNO", "Hydro", "Bio Power- CNO"]

def safe_add_carrier(n, name, **kw):
    """Adds a carrier if it doesn't already exist."""
    if name not in n.carriers.index:
        n.add("Carrier", name, **kw)

# --- calculate_metrics function (for summary_solar_wind_ls.csv) ---
def calculate_metrics(n, generator_type, generator_index_name=None):
    """Calculate performance metrics for given generator type."""
    # Ensure n.generators_t.p is not empty
    if n.generators_t.p.empty:
        return 0, 0, 0, 0, 0, 0, 0, 0

    gen_indices = n.generators[n.generators.carrier == generator_type].index
    if gen_indices.empty:
        return 0, 0, 0, 0, 0, 0, 0, 0

    # Ensure intersection with actual dispatch columns
    dispatch_cols = gen_indices.intersection(n.generators_t.p.columns)
    if dispatch_cols.empty:
        return 0, 0, 0, 0, 0, 0, 0, 0

    generation = n.generators_t.p[dispatch_cols].sum().sum()
    capacity = n.generators.loc[dispatch_cols, 'p_nom_opt'].sum()

    CUF = generation / (capacity * 8760) if capacity > 0 else 0

    p_max_pu_series = pd.Series(1.0, index=n.generators_t.p.index)
    if not n.generators_t.p_max_pu.empty and generator_index_name and generator_index_name in n.generators_t.p_max_pu.columns:
        p_max_pu_series = n.generators_t.p_max_pu[generator_index_name]
    elif not n.generators_t.p_max_pu.empty and not dispatch_cols.empty:
        p_max_pu_series = n.generators_t.p_max_pu[dispatch_cols].mean(axis=1)


    max_generation_potential = (p_max_pu_series * capacity).sum()
    
    percentage_generation_from_max = (generation / max_generation_potential) if max_generation_potential > 0 else 0
    curtailment = (max_generation_potential - generation) if max_generation_potential > 0 else 0
    max_generation = max_generation_potential

    slack_contribution = 0
    peak_shortage = 0
    if 'slack' in n.generators.carrier.unique():
        slack_gen_indices = n.generators[n.generators.carrier == 'slack'].index.intersection(n.generators_t.p.columns)
        if not slack_gen_indices.empty:
            total_slack_generation = n.generators_t.p[slack_gen_indices].sum().sum()
            total_load = n.loads_t.p.sum().sum() if not n.loads_t.p.empty else 0
            slack_contribution = (total_slack_generation / total_load) if total_load > 0 else 0
            peak_shortage = n.generators_t.p[slack_gen_indices].sum(axis=1).max()

    return generation, capacity, CUF, percentage_generation_from_max, curtailment, max_generation, slack_contribution, peak_shortage

# --- create_plots function (for HTML outputs) ---
def create_plots(n, run_folder, scenario_name, scenario_year): # Now accepts scenario_name and scenario_year
    """
    Create HTML plots from the network `n`.
    Saves HTML files to run_folder / "plots".
    """
    plot_folder = os.path.join(run_folder, "plots")
    os.makedirs(plot_folder, exist_ok=True)

    safe_scenario_name = re.sub(r'[^\w\-_\.]', '_', scenario_name)

    A = pd.DataFrame()
    try:
        battery_links_indices = n.links[n.links.carrier == 'battery_link'].index
        if not battery_links_indices.empty and hasattr(n.links_t, 'p') and not n.links_t.p.empty:
             A = n.links_t.p[battery_links_indices]
        else:
             print("DEBUG: No battery links found or no time series data for them.")
    except Exception as e:
        print(f"DEBUG: Error accessing battery link time series: {e}")
        A = pd.DataFrame()


    # (1) Capacity plot - Optimal Generation Capacity (dropping slack)
    try:
        generators = n.generators.copy()
        if not generators.empty:
            df_capacity_gen = generators.groupby(by='carrier').p_nom_opt.sum().reset_index(name='optimal_capacity')
            
            if 'slack' in df_capacity_gen['carrier'].values:
                df_capacity_gen = df_capacity_gen[df_capacity_gen['carrier'] != 'slack']

            stores = n.stores
            if not stores.empty:
                df_capacity_store = stores.groupby(by='carrier').e_nom_opt.sum().reset_index(name='optimal_capacity') if 'carrier' in stores.columns else pd.DataFrame([{'carrier': 'Battery Storage', 'optimal_capacity': stores.e_nom_opt.sum()}])
                df_capacity_store['carrier'] = df_capacity_store['carrier'].apply(lambda x: str(x).title() + ' Storage' if str(x).lower() != 'battery storage' else x)
                df_capacity_combined = pd.concat([df_capacity_gen, df_capacity_store], ignore_index=True)
            else:
                df_capacity_combined = df_capacity_gen

            df_capacity_combined.to_csv(os.path.join(run_folder, f"capacity_data_{scenario_name}.csv"), index=False)

            fig = px.bar(df_capacity_combined,
                         x='carrier',
                         y='optimal_capacity',
                         title=f'Optimal Capacity by Carrier - Scenario {scenario_name}',
                         labels={'optimal_capacity': 'Capacity (MW/MWh)', 'carrier': 'Carrier'},
                         color='carrier', # Color by carrier for distinction
                         barmode='group')
            fig.update_layout(xaxis_title='Carrier', yaxis_title='Capacity (MW)/ Capacity-Battery (MWh)', template='simple_white')
            outfn = os.path.join(plot_folder, f"Capacity_Scenario_{safe_scenario_name}.html")
            fig.write_html(outfn)
        else:
            print("Capacity plot skipped: No generator data.")
    except Exception as e:
        print(f"Capacity plot failed: {e}")


    # (3) Dispatch plot - stacked generation by carrier + demand + storage flows
    try:
        if (hasattr(n, 'generators_t') and hasattr(n.generators_t, 'p') and n.generators_t.p.empty) and \
           (hasattr(n, 'loads_t') and hasattr(n.loads_t, 'p_set') and n.loads_t.p_set.empty):
            print("Dispatch plot skipped: No generator or load dispatch data.")
            raise ValueError("No dispatch or load data for plotting.")

        colours = {
            'diesel': 'brown', 'solar': 'gold', 'wind': 'darkgreen', 'hydro': 'blue',
            'slack': 'grey', 'gas': 'darkorange', 'cno': '#6A3D9A', 'solar rooftop': 'lightgoldenrodyellow',
            'geothermal': '#CAB2D6', 'bio power- cno': '#6A3D9A',
            'battery storage': 'purple',
            'charge': 'darkred', 'discharge': 'limegreen'
        }

        fig = go.Figure()
        
        df_gen_dispatch_for_plot = pd.DataFrame(index=n.snapshots)
        if not n.generators_t.p.empty:
            df_gen_dispatch_agg = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum()
            
            if 'slack' in df_gen_dispatch_agg.columns:
                df_gen_dispatch_for_plot = df_gen_dispatch_agg.drop(columns=['slack'])
            else:
                df_gen_dispatch_for_plot = df_gen_dispatch_agg
        
        plot_order_carriers = [c for c in ['Hydro', 'Wind', 'Solar', 'Solar Rooftop', 'Geothermal', 'Bio Power- CNO', 'Gas', 'CNO', 'Diesel'] if c in df_gen_dispatch_for_plot.columns]
        other_carriers = [c for c in df_gen_dispatch_for_plot.columns if c not in plot_order_carriers]
        final_plot_carriers_order = plot_order_carriers + sorted(other_carriers)

        for carr in final_plot_carriers_order:
            if carr in df_gen_dispatch_for_plot.columns and not df_gen_dispatch_for_plot[carr].empty:
                color = colours.get(str(carr).lower(), 'lightgrey')
                fig.add_trace(go.Scatter(
                    x=df_gen_dispatch_for_plot.index,
                    y=df_gen_dispatch_for_plot[carr],
                    stackgroup='generation',
                    name=carr,
                    mode='lines',
                    line=dict(width=0.5, color=color),
                    fill='tozeroy',
                    fillcolor=color,
                    hoverinfo='x+y+name'
                ))

        if not A.empty:
            y_charge_sum = A.where(A < 0, 0).sum(axis=1).abs().reindex(n.snapshots, fill_value=0)
            y_discharge_sum = A.where(A > 0, 0).sum(axis=1).reindex(n.snapshots, fill_value=0)

            if not y_discharge_sum.empty and y_discharge_sum.sum() > 0:
                fig.add_trace(go.Scatter(
                    x=y_discharge_sum.index,
                    y=y_discharge_sum,
                    stackgroup='generation',
                    name='Storage Discharge',
                    mode='lines',
                    line=dict(width=0.5, color=colours.get('discharge', 'darkgreen')),
                    fill='tozeroy',
                    fillcolor=colours.get('discharge', 'darkgreen'),
                    hoverinfo='x+y+name'
                ))
            
            if hasattr(n, 'loads_t') and hasattr(n.loads_t, 'p_set') and not n.loads_t.p_set.empty:
                total_demand = n.loads_t.p_set.sum(axis=1)

                fig.add_trace(go.Scatter(
                    x=total_demand.index,
                    y=total_demand,
                    mode='lines',
                    name='Demand',
                    line=dict(color='black', width=2, dash='dot'),
                    hoverinfo='x+y+name'
                ))

                if not y_charge_sum.empty and y_charge_sum.sum() > 0:
                    fig.add_trace(go.Scatter(
                        x=y_charge_sum.index,
                        y=-y_charge_sum,
                        mode='lines',
                        name='Storage Charge',
                        line=dict(width=0.5, color=colours.get('charge', 'darkred')),
                        fill='tozeroy',
                        fillcolor=colours.get('charge', 'darkred'),
                        hoverinfo='x+y+name'
                    ))
        
        dispatch_df_to_save = pd.DataFrame(index=n.snapshots)
        
        if hasattr(n, 'loads_t') and hasattr(n.loads_t, 'p_set') and not n.loads_t.p_set.empty:
            dispatch_df_to_save['Demand'] = n.loads_t.p_set.sum(axis=1)
        
        if not n.generators_t.p.empty:
            df_gen_dispatch_full = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum()
            for carr in df_gen_dispatch_full.columns:
                dispatch_df_to_save[f'Gen_{carr}'] = df_gen_dispatch_full[carr]
        
        if not A.empty:
            dispatch_df_to_save['Storage_Charge'] = A.where(A < 0, 0).sum(axis=1).abs().reindex(n.snapshots, fill_value=0)
            dispatch_df_to_save['Storage_Discharge'] = A.where(A > 0, 0).sum(axis=1).reindex(n.snapshots, fill_value=0)

        if not dispatch_df_to_save.empty:
            dispatch_df_to_save.to_csv(os.path.join(run_folder, f"dispatch_data_{scenario_name}.csv"))


        fig.update_layout(
            title=f'Generation Dispatch - Scenario {scenario_name}',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Power (MW)'),
            template='simple_white',
            hovermode='x unified',
            yaxis_range=[-(total_demand.max() * 0.5) if (hasattr(n, 'loads_t') and hasattr(n.loads_t, 'p_set') and not n.loads_t.p_set.empty) else -10, (total_demand.max() * 1.5) if (hasattr(n, 'loads_t') and hasattr(n.loads_t, 'p_set') and not n.loads_t.p_set.empty) else 100]
        )
        fig.write_html(os.path.join(plot_folder, f"Generation_Dispatch_Scenario_{safe_scenario_name}.html"))
    except Exception as e:
        print(f"Dispatch plot failed: {e}")


    # (4) System cost stacked bar (Fixed Costs, Variable Costs, Storage Costs, and LCOE)
    try:
        if n.objective is not None and not n.generators.empty and not n.loads_t.p_set.empty:
            total_annual_demand_MWh = n.loads_t.p_set.sum().sum()
            
            gen_capital_cost_annual = (n.generators.capital_cost * n.generators.p_nom_opt).sum() if not n.generators.empty else 0
            gen_fixed_operation_cost_annual = (n.generators.fixed_operation_cost * n.generators.p_nom_opt).sum() if 'fixed_operation_cost' in n.generators.columns and not n.generators.empty else 0

            gen_variable_costs_annual = (n.generators_t.p * n.generators.marginal_cost).sum().sum() if not n.generators_t.p.empty else 0

            store_capital_cost_annual = (n.stores.capital_cost * n.stores.e_nom_opt).sum() if not n.stores.empty else 0
            link_storage_capital_cost_annual = (n.links.capital_cost * n.links.p_nom_opt).loc[n.links.carrier == 'battery_link'].sum() if not n.links.empty else 0
            total_storage_fixed_cost_annual = store_capital_cost_annual + link_storage_capital_cost_annual
            
            line_capital_cost_annual = (n.lines.capital_cost * n.lines.s_nom_opt).sum() if not n.lines.empty else 0
            transformer_capital_cost_annual = (n.transformers.capital_cost * n.transformers.s_nom).sum() if not n.transformers.empty else 0
            total_transmission_fixed_cost_annual = line_capital_cost_annual + transformer_capital_cost_annual

            slack_cost_annual = (n.generators_t.p['slack'] * n.generators.loc['slack', 'marginal_cost']).sum() if 'slack' in n.generators.index and not n.generators_t.p.empty else 0
            
            
            cost_data_breakdown = {
                'Cost Type': [],
                'Amount (USD/year)': []
            }
            if gen_capital_cost_annual > 1e-3: cost_data_breakdown['Cost Type'].append('Generator Capital'); cost_data_breakdown['Amount (USD/year)'].append(gen_capital_cost_annual)
            if gen_fixed_operation_cost_annual > 1e-3: cost_data_breakdown['Cost Type'].append('Generator Fixed O&M'); cost_data_breakdown['Amount (USD/year)'].append(gen_fixed_operation_cost_annual)
            if gen_variable_costs_annual > 1e-3: cost_data_breakdown['Cost Type'].append('Generator Variable'); cost_data_breakdown['Amount (USD/year)'].append(gen_variable_costs_annual)
            if total_storage_fixed_cost_annual > 1e-3: cost_data_breakdown['Cost Type'].append('Storage Fixed'); cost_data_breakdown['Amount (USD/year)'].append(total_storage_fixed_cost_annual)
            if total_transmission_fixed_cost_annual > 1e-3: cost_data_breakdown['Cost Type'].append('Transmission Fixed'); cost_data_breakdown['Amount (USD/year)'].append(total_transmission_fixed_cost_annual)
            if slack_cost_annual > 1e-3: cost_data_breakdown['Cost Type'].append('Unserved Energy'); cost_data_breakdown['Amount (USD/year)'].append(slack_cost_annual)
            
            df_costs = pd.DataFrame(cost_data_breakdown)

            LCOE = np.nan
            if not df_costs.empty and total_annual_demand_MWh > 0:
                total_annual_cost = df_costs['Amount (USD/year)'].sum()
                LCOE = total_annual_cost / total_annual_demand_MWh
            
            df_costs.to_csv(os.path.join(run_folder, f"system_cost_data_{scenario_name}.csv"), index=False)

            fig = px.bar(df_costs, x='Cost Type', y='Amount (USD/year)',
                         title=f'Annual System Cost Breakdown - Scenario {scenario_name}<br><sup>LCOE: {LCOE:.2f} USD/MWh</sup>',
                         labels={'Amount (USD/year)': 'Amount (USD/year)', 'Cost Type': 'Cost Category'},
                         color='Cost Type')
            fig.update_layout(xaxis_title='Cost Category', yaxis_title='Annual Cost (USD/year)', template='simple_white')
            fig.write_html(os.path.join(plot_folder, f"System_Costs_Scenario_{safe_scenario_name}.html"))

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculated LCOE: {LCOE:.2f} USD/MWh")

        else:
            print("System cost plot skipped: No objective value or generator/load data for cost calculations.")
    except Exception as e:
        print(f"System cost plot failed: {e}")
    print(f"Plots created (if data available) in {plot_folder}")


# --------------------------
# Comparison Tab Helper Functions (REFINED for numerical-only)
# --------------------------

def load_network_from_nc(file_path):
    """Loads a PyPSA network from a NetCDF file."""
    try:
        n = pypsa.Network(file_path)
        return n
    except Exception as e:
        raise ValueError(f"Failed to load network from {file_path}: {e}")

def extract_key_metrics(n, scenario_name):
    """Extracts a predefined set of key scalar metrics from a PyPSA network."""
    metrics = {
        'Scenario': scenario_name,
        'Total System Cost (USD)': n.objective if n.objective is not None else np.nan,

        # Installed Capacity
        'Total Generation Capacity (MW)': n.generators.p_nom_opt.sum() if not n.generators.empty else 0,
        'Total Storage Capacity (MWh)': n.stores.e_nom_opt.sum() if not n.stores.empty else 0,
        'Total Line Capacity (MVA)': n.lines.s_nom_opt.sum() if not n.lines.empty else 0,

        # Annual Generation
        'Total Annual Generation (GWh)': n.generators_t.p.sum().sum() / 1e3 if not n.generators_t.p.empty else 0,
        'Total Annual Renewable Generation (GWh)': n.generators_t.p.loc[:, n.generators.carrier.isin(get_renewable_carriers())].sum().sum() / 1e3 if not n.generators_t.p.empty else 0,
        'Total Annual Demand (GWh)': n.loads_t.p_set.sum().sum() / 1e3 if not n.loads_t.p_set.empty else 0,

        # Emissions & RE Share
        'Total Annual CO2 Emissions (tons)': (n.generators_t.p.sum().groupby(n.generators.carrier).sum() * n.carriers.co2_emissions).sum() if not n.generators_t.p.empty and 'co2_emissions' in n.carriers.columns else 0,
    }
    
    # Calculate RE Share
    if metrics['Total Annual Demand (GWh)'] > 0:
        metrics['Achieved RE Share (%)'] = (metrics['Total Annual Renewable Generation (GWh)'] / metrics['Total Annual Demand (GWh)']) * 100
    else:
        metrics['Achieved RE Share (%)'] = 0

    # Calculate LCOE
    LCOE = np.nan
    if not np.isnan(metrics['Total System Cost (USD)']) and metrics['Total Annual Generation (GWh)'] > 0:
        LCOE = (metrics['Total System Cost (USD)'] / (metrics['Total Annual Generation (GWh)'] * 1000))
    metrics['LCOE (USD/MWh)'] = LCOE
    

    # Add capacity by carrier breakdown
    if not n.generators.empty:
        cap_by_carrier = n.generators.groupby('carrier').p_nom_opt.sum()
        for carrier, capacity in cap_by_carrier.items():
            metrics[f'Capacity {carrier.title()} (MW)'] = capacity
    
    # Add generation by carrier breakdown
    if not n.generators_t.p.empty:
        gen_by_carrier = n.generators_t.p.sum().groupby(n.generators.carrier).sum() / 1e3
        for carrier, generation in gen_by_carrier.items():
            metrics[f'Generation {carrier.title()} (GWh)'] = generation

    return metrics

# --------------------------
# Core model runner
# --------------------------
def run_model(
    data_file, # Expected to be io.BytesIO buffer from Streamlit
    results_dir,
    solver,
    co2_cap,
    re_share,
    slack_cost,
    discount_rate,
    tech_cost_multipliers,
    scenario_name,
    scenario_number,
    line_expansion,
    enabled_techs,
    default_new_gen_extendable,
    scenario_year,
    target_peak_demand,
    df_buses,
    df_generators,
    df_load,
    df_transmission_lines,
    df_transformers,
    df_storage,
    df_generation_profiles,
    df_scenario_year,
    demand_projection_method,
    demand_growth_percentage
):
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Starting simulation for scenario: {scenario_name}"
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Results will be saved to: {results_dir}"

    os.makedirs(results_dir, exist_ok=True)

    if solver.lower() == 'highs' and os.environ.get('PYPSA_SOLVER_HIGHSPY_PATH_SET') == 'true':
        yield f"[{datetime.now().strftime('%H:%M:%S')}] HiGHS executable path was successfully added to environment PATH by app startup."
    elif solver.lower() == 'highs':
         yield f"[{datetime.now().strftime('%H:%M:%S')}] Attempting to find HiGHS using system PATH (automatic detection in app.py failed)."


    # --------------------------
    # Scenario Year and Demand Scaling
    # --------------------------
    target_year = scenario_year # Use scenario_year directly from GUI input
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Target Year for simulation: {target_year}"

    # Load Demand_C_TS_ from the provided data_file buffer
    data_file.seek(0) 
    load_raw_df = pd.read_excel(data_file, sheet_name='Load_C_TS_')
    
    # Compute current peak and target peak for scaling
    yearly_load_curve_base = load_raw_df.loc[:, load_raw_df.columns != "Total"] 
    current_peak_MW = yearly_load_curve_base.sum(axis=1).max() # Max sum of hourly demand over all load centers
    
    scale_factor = 1.0
    if demand_projection_method == "Target Peak Demand":
        target_peak_MW_input = target_peak_demand
        scale_factor = target_peak_MW_input / current_peak_MW if current_peak_MW > 0 else 1.0
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Demand Scaling Method: Target Peak. Base Load Peak: {current_peak_MW:.2f} MW, Target Peak: {target_peak_MW_input:.2f} MW. Scale Factor: {scale_factor:.2f}"
    else: # Percentage Growth
        growth_factor_percentage = demand_growth_percentage / 100.0
        scale_factor = (1 + growth_factor_percentage)
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Demand Scaling Method: Percentage Growth. Percentage Growth: {demand_growth_percentage:.1f}%. Scale Factor: {scale_factor:.2f}"

    if current_peak_MW == 0:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Base load peak is zero, demand scaling skipped."

    # --------------------------
    # Network build
    # --------------------------
    n = pypsa.Network()

    # Snapshots - with leap year handling
    timestamps = pd.date_range(start=f'{target_year}-01-01', end=f'{target_year}-12-31 23:00', freq='h')
    if calendar.isleap(target_year) and len(timestamps) == 8784: # If it's a leap year and has 8784 hours
        feb29_start = pd.Timestamp(f'{target_year}-02-29 00:00')
        feb29_end = pd.Timestamp(f'{target_year}-02-29 23:00')
        timestamps = timestamps[~((timestamps >= feb29_start) & (timestamps <= feb29_end))]
    elif not calendar.isleap(target_year) and len(timestamps) > 8760:
         timestamps = timestamps[:8760] # Trim to 8760
    elif len(timestamps) < 8760:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Snapshots generated fewer than 8760 hours ({len(timestamps)}). This may affect results."
    
    n.snapshots = timestamps
    n.snapshot_weightings['objective'] = n.snapshot_weightings['objective'] * (8760 / len(timestamps)) # Adjust weighting if snapshot length not 8760
    yield f"[{datetime.now().strftime('%H:%M:%S')}] PyPSA Network initialized with {len(timestamps)} snapshots (leap year adjusted)."

    # Buses - ADDED: v_nom, carrier, unit
    if not df_buses.empty:
        df_buses_processed = df_buses.set_index('Bus name', drop=False).copy()
        for bus_name, row in df_buses_processed.iterrows():
            n.add("Bus",
                bus_name,
                x=row.get("x"),
                y=row.get("y"),
                v_nom=row.get("v_nom"),
                carrier=row.get("carrier"),
                unit=row.get("unit"))
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Added {len(df_buses_processed)} buses with extended attributes."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No bus data provided."
        raise ValueError("No bus data provided to build the network.")

    # Carriers - Ensure consistent names and co2 emissions
    all_carriers_from_generators = df_generators['Carrier'].dropna().astype(str).unique()
    all_carriers_from_storage = df_storage['Carrier'].dropna().astype(str).unique()
    all_carriers_from_buses = df_buses['carrier'].dropna().astype(str).unique()
    all_carriers_from_data = set(all_carriers_from_generators).union(set(all_carriers_from_storage)).union(set(all_carriers_from_buses))
    
    standard_carriers = {"electricity", "backup", "AC", "storage_charge", "storage_discharge", "slack", "battery_link"}
    all_carriers_to_add = all_carriers_from_data.union(standard_carriers)

    for c in all_carriers_to_add:
        carrier_name_lower = str(c).lower() 
        if carrier_name_lower == "diesel":
            safe_add_carrier(n, c, co2_emissions=0.267)
        elif carrier_name_lower == "gas":
            safe_add_carrier(n, c, co2_emissions=0.202)
        elif carrier_name_lower == "bio power- cno":
            safe_add_carrier(n, c, co2_emissions=0.1)
        else:
            safe_add_carrier(n, c, co2_emissions=0.0)
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Defined {len(n.carriers)} energy carriers."

    # Demand - Apply new scaling factor
    if not df_load.empty:
        df_load_processed = df_load.copy()
        
        if 'Total' in df_load_processed.columns:
            df_load_processed = df_load_processed.drop(columns=['Total'])
        
        if not isinstance(df_load_processed.index, pd.DatetimeIndex):
            if len(df_load_processed) == len(n.snapshots):
                df_load_processed.index = n.snapshots
            else:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Load data rows ({len(df_load_processed)}) do not match snapshot count ({len(n.snapshots)}). Reindexing will occur."
                df_load_processed = df_load_processed.reindex(n.snapshots)
        
        df_load_processed = df_load_processed.fillna(0)

        numeric_cols = df_load_processed.select_dtypes(include=np.number).columns
        df_load_processed = df_load_processed[numeric_cols]
        df_load_processed.dropna(axis=1, how='all', inplace=True)

        if df_load_processed.empty:
            yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Processed load data is empty or non-numeric. No loads added to network."
        else:
            for load_centre in df_load_processed.columns:
                if load_centre in n.buses.index:
                    load_fix = pd.Series(df_load_processed[load_centre] * scale_factor, index=n.snapshots, name=load_centre)
                    n.add("Load", load_centre, bus=load_centre, p_set=load_fix)
                else:
                    yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Load center '{load_centre}' does not have a corresponding bus. Skipping."
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Added {len(n.loads)} load components, scaled by {scale_factor:.2f}."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No load data provided."


    # Process Generation Profiles (ensure they are Series with n.snapshots as index)
    # Initialize all profiles to 0.0, will be overwritten if data is present and tech enabled
    W_gen_profile = pd.Series(0.0, index=n.snapshots)
    S_gen_profile = pd.Series(0.0, index=n.snapshots)
    SR_gen_profile = pd.Series(0.0, index=n.snapshots) # NEW: For Solar Rooftop
    H_gen_profile = pd.Series(0.0, index=n.snapshots)

    if not df_generation_profiles.empty:
        # Determine if df_generation_profiles is a dict (from Excel mapping) or a DataFrame (from Manual entry)
        if isinstance(df_generation_profiles, dict) and 'df_content' in df_generation_profiles:
            # Reconstruct DataFrame from dict of lists, using snapshots as index
            df_profiles_content = df_generation_profiles['df_content']
            # Filter out None values that indicate missing profiles from mapping
            valid_profiles_content = {k: v for k, v in df_profiles_content.items() if v is not None}
            if valid_profiles_content:
                # Ensure all lists are of the correct length, or pad with 0.0
                for k, v in valid_profiles_content.items():
                    if len(v) < len(n.snapshots):
                        valid_profiles_content[k] = v + [0.0] * (len(n.snapshots) - len(v))
                    elif len(v) > len(n.snapshots):
                        valid_profiles_content[k] = v[:len(n.snapshots)]

                df_gen_profiles_from_map = pd.DataFrame(valid_profiles_content, index=n.snapshots)
            else:
                df_gen_profiles_from_map = pd.DataFrame(index=n.snapshots)
        else: # If using Manual Entry or direct DataFrame, it's already a DataFrame
            df_gen_profiles_from_map = df_generation_profiles.copy()
            if not isinstance(df_gen_profiles_from_map.index, pd.DatetimeIndex):
                if len(df_gen_profiles_from_map) == len(n.snapshots):
                    df_gen_profiles_from_map.index = n.snapshots
                else:
                    yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Generation profile rows ({len(df_gen_profiles_from_map)}) do not match snapshot count ({len(n.snapshots)}). Reindexing will occur."
                    df_gen_profiles_from_map = df_gen_profiles_from_map.reindex(n.snapshots).fillna(0.0)
            else:
                df_gen_profiles_from_map = df_gen_profiles_from_map.reindex(n.snapshots).fillna(0.0).copy()

        # Load profiles only if tech is enabled AND column exists in the processed DataFrame
        if enabled_techs.get('Wind', False) and 'Wind profile' in df_gen_profiles_from_map.columns:
            W_gen_profile = pd.to_numeric(df_gen_profiles_from_map['Wind profile'], errors='coerce').fillna(0.0)
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Wind generation profile loaded."
        elif enabled_techs.get('Wind', False):
            yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Wind is enabled but 'Wind profile' data is missing or not mapped. Wind generators will produce 0 power."

        if enabled_techs.get('Solar', False) and 'Solar profile' in df_gen_profiles_from_map.columns:
            S_gen_profile = pd.to_numeric(df_gen_profiles_from_map['Solar profile'], errors='coerce').fillna(0.0)
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Solar generation profile loaded."
        elif enabled_techs.get('Solar', False):
            yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Solar is enabled but 'Solar profile' data is missing or not mapped. Solar generators will produce 0 power."
        
        # NEW: Load Solar Rooftop profile separately
        if enabled_techs.get('Solar Rooftop', False) and 'Solar Rooftop profile' in df_gen_profiles_from_map.columns:
            SR_gen_profile = pd.to_numeric(df_gen_profiles_from_map['Solar Rooftop profile'], errors='coerce').fillna(0.0)
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Solar Rooftop generation profile loaded."
        elif enabled_techs.get('Solar Rooftop', False):
            yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Solar Rooftop is enabled but 'Solar Rooftop profile' data is missing or not mapped. Solar Rooftop generators will produce 0 power."

        if enabled_techs.get('Hydro', False) and 'Hydro profile' in df_gen_profiles_from_map.columns:
            H_gen_profile = pd.to_numeric(df_gen_profiles_from_map['Hydro profile'], errors='coerce').fillna(0.0)
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Hydro generation profile loaded."
        elif enabled_techs.get('Hydro', False):
            yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Hydro is enabled but 'Hydro profile' data is missing or not mapped. Hydro generators will produce 0 power."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No generation profile data provided. Intermittent generators will produce 0 power."


    # --------------------------
    # Generators
    # --------------------------
    if not df_generators.empty:
        df_generators_processed = df_generators.set_index('Generator name', drop=True).copy()
        
        if 'Scenario' in df_generators_processed.columns:
             df_generators_processed['Scenario'] = df_generators_processed['Scenario'].apply(
                 lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x
             )

        generators_for_scenario = df_generators_processed[
            df_generators_processed['Scenario'].apply(lambda x: scenario_number in x if isinstance(x, list) else False)
        ].copy()

        added_generators_count = 0
        for gen_i, row in generators_for_scenario.iterrows():
            carrier = row['Carrier']
            if not enabled_techs.get(carrier, True):
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Skipping generator '{gen_i}' (Carrier: {carrier}) as its technology is disabled."
                continue

            bus = row['Bus']
            if bus not in n.buses.index:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Generator '{gen_i}' refers to non-existent bus '{bus}'. Skipping."
                continue

            status = row.get('Status', 0) # 0 for existing, 1 for new
            p_nom_initial = float(row.get('Capacity(MW)', 0.0)) if status == 0 else 0.0

            # --- CORRECTED: Robust p_nom_extendable parsing ---
            input_p_nom_extendable_raw = row.get('p_nom_extendable')
            
            # Ensure input_p_nom_extendable_raw is a scalar for pd.notna
            if isinstance(input_p_nom_extendable_raw, (pd.Series, np.ndarray, list)):
                if len(input_p_nom_extendable_raw) > 0:
                    # Attempt to get first value, convert to string then bool
                    val = str(input_p_nom_extendable_raw.iloc[0] if isinstance(input_p_nom_extendable_raw, pd.Series) else input_p_nom_extendable_raw[0]).lower().strip()
                    p_nom_extendable = (val == 'true' or val == '1')
                else:
                    p_nom_extendable = False # Treat empty series/list as not extendable
            elif input_p_nom_extendable_raw is not None and pd.notna(input_p_nom_extendable_raw):
                p_nom_extendable = bool(input_p_nom_extendable_raw)
            elif status == 1: # If New (Status=1) and not explicitly specified, use GUI default
                p_nom_extendable = default_new_gen_extendable
            else: # status == 0 (Existing) and not explicitly specified: default to False
                p_nom_extendable = False
            # --- END CORRECTED ---

            # --- Robust conversion for numerical inputs ---
            lifetime = int(float(row.get('lifetime', 25)))
            raw_capex_per_MW = float(row.get('Capital_cost (USD/MW)', 0.0))
            fixed_OM_per_MW_year = float(row.get('fixed_O&M (USD/MW)', 0.0))
            marginal_cost_per_MWh = float(row.get('Marginal cost (USD/MWh)', 0.0))
            
            p_min_pu = float(row.get('min_generation_level', 0.0))

            # --- HYDRO PROFILE FIX ---
            # As per main.py, for "run-of-river" hydro, p_min_pu must equal p_max_pu.
            if carrier == 'Hydro':
                p_min_pu = H_gen_profile # Override p_min_pu with the hydro profile
            # --- END HYDRO PROFILE FIX ---

            efficiency = float(row.get('efficiency', 1.0))
            
            committable_raw = row.get('committable', False) # Boolean input
            committable = bool(committable_raw) if pd.notna(committable_raw) else False # Robust bool conversion
            
            # --- CORRECTED: Generator Capital Cost (only Capital_cost (USD/MW), Fixed O&M NOT bundled for annuity) ---
            annuitized_capex = calculate_annuity(raw_capex_per_MW, discount_rate, lifetime)

            p_max_pu_value = 1.0
            if carrier == 'Wind':
                p_max_pu_value = W_gen_profile
            elif carrier == 'Solar': # Separated from Solar Rooftop
                p_max_pu_value = S_gen_profile
            elif carrier == 'Solar Rooftop': # NEW: Separate profile for Solar Rooftop
                p_max_pu_value = SR_gen_profile
            elif carrier == 'Hydro':
                p_max_pu_value = H_gen_profile

            qty = int(float(row.get('Quantity', 1.0))) if pd.notna(row.get('Quantity', 1.0)) else 1 # CORRECTED: Robust int conversion for Quantity
            size_mw = float(row.get('Size (MW)', p_nom_initial)) # Default to p_nom_initial if Size not specified
            
            p_nom_min_gen = float(row.get('P_nom_min', 0.0))
            p_nom_max_gen = float(row.get('P_nom_max', np.inf)) # Default to infinity for max


            if status == 0 and qty > 1 and pd.notna(size_mw) and size_mw > 0:
                for i in range(1, qty + 1):
                    gen_name_instance = f"{gen_i}_{carrier[:2].upper()}{i}"
                    if gen_name_instance in n.generators.index:
                        gen_name_instance = f"{gen_i}_{carrier[:2].upper()}{i}_{added_generators_count}"
                    n.add("Generator",
                        gen_name_instance,
                        bus=bus,
                        p_nom=size_mw, # Capacity of individual unit
                        p_nom_min=p_nom_min_gen,
                        p_nom_max=p_nom_max_gen,
                        p_min_pu=p_min_pu,
                        p_max_pu=p_max_pu_value,
                        carrier=carrier,
                        efficiency=efficiency,
                        marginal_cost=marginal_cost_per_MWh,
                        capital_cost=annuitized_capex, # CORRECTED: Ensured annuitized_capex is used
                        p_nom_extendable=p_nom_extendable,
                        committable=committable,
                        ramp_limit_down=1, ramp_limit_up=1,
                        p_nom_initial=p_nom_initial, # Store initial p_nom as custom attribute
                        fixed_operation_cost=fixed_OM_per_MW_year * size_mw # NEW: add fixed O&M explicitly
                    )
                    added_generators_count += 1
                    yield f"[{datetime.now().strftime('%H:%M:%S')}] Added Existing Gen '{gen_name_instance}' ({carrier}) at {bus}, p_nom={size_mw:.2f}, marginal_cost={marginal_cost_per_MWh:.2f} USD/MWh."
            else:
                n.add("Generator",
                    gen_i,
                    bus=bus,
                    p_nom=p_nom_initial, # Total initial capacity of plant
                    p_nom_min=p_nom_min_gen,
                    p_nom_max=p_nom_max_gen,
                    p_min_pu=p_min_pu,
                    p_max_pu=p_max_pu_value,
                    carrier=carrier,
                    efficiency=efficiency,
                    marginal_cost=marginal_cost_per_MWh,
                    capital_cost=annuitized_capex, # CORRECTED: Ensured annuitized_capex is used
                    p_nom_extendable=p_nom_extendable,
                    committable=committable,
                    ramp_limit_down=1, ramp_limit_up=1,
                    p_nom_initial=p_nom_initial, # Store initial p_nom as custom attribute
                    fixed_operation_cost=fixed_OM_per_MW_year * p_nom_initial if p_nom_initial > 0 else 0 # NEW: add fixed O&M explicitly
                )
                added_generators_count += 1
                status_str = "New" if status == 1 else "Existing" 
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Added {status_str} Gen '{gen_i}' ({carrier}) at {bus}, p_nom={p_nom_initial:.2f}, marginal_cost={marginal_cost_per_MWh:.2f} USD/MWh, extendable={p_nom_extendable}."

        slack_bus = list(n.buses.index)[0] if not n.buses.empty else "DummyBusForSlack" # Use first bus or dummy
        if slack_bus not in n.buses.index: # If dummy, ensure it exists
            n.add("Bus", slack_bus)

        n.add("Generator",
                'slack',
                bus=slack_bus,
                p_nom=1e6, # Large nominal capacity
                p_max_pu=1,
                marginal_cost=slack_cost,
                carrier='slack')
        added_generators_count += 1
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Added slack generator at bus '{slack_bus}' with marginal_cost={slack_cost:.2f} USD/MWh."
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Total {added_generators_count} generators added."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No generator data provided."

    # --------------------------
    # Transmission Lines
    # --------------------------
    if not df_transmission_lines.empty:
        df_transmission_lines_processed = df_transmission_lines.copy()
        df_transmission_lines_processed.rename(columns={
            'Capital_cost (USD/MVA)': 'capital_cost',
            'Length (kM)': 'length',
            'From': 'bus0',
            'To': 'bus1'
        }, inplace=True)

        for i, row in df_transmission_lines_processed.iterrows():
            from_bus = row['bus0']
            to_bus = row['bus1']
            if from_bus not in n.buses.index or to_bus not in n.buses.index:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Transmission line between '{from_bus}' and '{to_bus}' refers to non-existent bus(es). Skipping."
                continue

            name = f"Line_{from_bus}_{to_bus}_{i}"
            s_nom_extendable = bool(row.get('s_nom_extendable', line_expansion))
            
            line_lifetime = int(float(row.get('lifetime', 25))) if pd.notna(row.get('lifetime',25)) else 25 # Robust conversion
            line_capital_cost_raw = float(row.get('capital_cost', 0.0))
            annuitized_line_capex = calculate_annuity(line_capital_cost_raw, discount_rate, line_lifetime)

            n.add("Line",
                  name,
                  bus0=from_bus,
                  bus1=to_bus,
                  type=row.get('type', None),
                  s_nom=float(row.get('s_nom', 0.0)) if not s_nom_extendable else 0.0,
                  s_nom_extendable=s_nom_extendable,
                  capital_cost=annuitized_line_capex,
                  length=float(row.get('length', 1.0)))
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Added Line '{name}': {from_bus} <-> {to_bus}, s_nom={row.get('s_nom',0.0):.2f}, extendable={s_nom_extendable}."
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Added {len(n.lines)} transmission lines."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No transmission line data provided."

    # --------------------------
    # Transformers - REMOVED: marginal_cost from input, not passed to n.add
    # --------------------------
    if not df_transformers.empty:
        df_transformers_processed = df_transformers.copy()
        df_transformers_processed.rename(columns={
            'Location': 'name',
            'Capital_cost (USD/MW)': 'capital_cost',
        }, inplace=True)

        for i, row in df_transformers_processed.iterrows():
            bus0 = str(row['bus0']).strip()
            bus1 = str(row['bus1']).strip()
            if bus0 not in n.buses.index or bus1 not in n.buses.index:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Transformer '{row.get('name', f'Trans_{i}')}' between '{bus0}' and '{bus1}' refers to non-existent bus(es). Skipping."
                continue
            
            name = str(row.get('name', f"Transformer_{i}")).strip()

            transformer_lifetime = int(float(row.get('lifetime', 25))) if pd.notna(row.get('lifetime', 25)) else 25 # Robust conversion
            transformer_capital_cost_raw = float(row.get('capital_cost', 0.0))
            annuitized_transformer_capex = calculate_annuity(transformer_capital_cost_raw, discount_rate, transformer_lifetime)
            
            n.add("Transformer",
                  name,
                  bus0=bus0,
                  bus1=bus1,
                  s_nom=float(row.get('s_nom', 0.0)),
                  v_nom0=float(row.get('v_nom0', 0.0)),
                  v_nom1=float(row.get('v_nom1', 0.0)),
                  x=float(row.get('x', 0.0)),
                  r=float(row.get('r', 0.0)),
                  capital_cost=annuitized_transformer_capex)
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Added Transformer '{name}': {bus0} <-> {bus1}, s_nom={row.get('s_nom', 0.0):.2f}."
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Added {len(n.transformers)} transformers."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No transformer data provided."


    # --------------------------
    # Storage construction (Revised to single bidirectional link model from main.py)
    # --------------------------
    if not df_storage.empty:
        df_storage_processed = df_storage.set_index('name', drop=True).copy()
        if 'Scenario' in df_storage_processed.columns:
             df_storage_processed['Scenario'] = df_storage_processed['Scenario'].apply(
                 lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x
             )

        storage_count = 0
        for sto_i, row in df_storage_processed.iterrows():
            bus_parent = row['Bus']
            if bus_parent not in n.buses.index:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Storage '{sto_i}' refers to non-existent parent bus '{bus_parent}'. Skipping."
                continue
            
            if 'Scenario' in row.index and not (scenario_number in row['Scenario'] if isinstance(row['Scenario'], list) else False):
                 yield f"[{datetime.now().strftime('%H:%M:%S')}] Skipping storage '{sto_i}' as it's not part of the current scenario."
                 continue
            
            carrier = row.get('Carrier', 'Battery Storage')
            if not enabled_techs.get(carrier, True):
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Skipping storage '{sto_i}' (Carrier: {carrier}) as its technology is disabled."
                continue

            # --- NEW Storage Model from main.py ---
            store_internal_bus_name = f"{sto_i}_internal_bus"
            if store_internal_bus_name not in n.buses.index:
                n.add("Bus", store_internal_bus_name)
                
            p_nom_converter = float(row.get('p_nom (MW)', 0.0))
            e_nom_storage = float(row.get('e_nom (MWh)', 0.0))
            status = float(row.get('Status', 0.0)) # Robust conversion
            
            e_nom_initial = e_nom_storage if status == 0 else 0.0
            
            e_nom_extendable = bool(row.get('e_nom_extendable', False))
            lifetime = int(float(row.get('lifetime', 20.0))) if pd.notna(row.get('lifetime', 20.0)) else 20 # Robust conversion
            
            marginal_cost_storage = float(row.get('Marginal cost (USD/MWh)', 0.0))
            raw_capex_per_MWh = float(row.get('Capital_cost (USD/MWh)', 0.0))
            
            annuitized_e_capex = calculate_annuity(raw_capex_per_MWh, discount_rate, lifetime)

            link_efficiency = float(row.get('link_efficiency', 0.95))
            link_marginal_cost = float(row.get('link_marginal_cost', 1.0))

            link_capital_cost_raw_per_MW = float(row.get('Capital_cost_Link (USD/MW)', 0.0))
            annuitized_link_capex = calculate_annuity(link_capital_cost_raw_per_MW, discount_rate, lifetime)

            e_max_pu = float(row.get('e_max_pu', 0.9))


            n.add(
                "Link", f"{sto_i}_link",
                bus0=store_internal_bus_name,
                bus1=bus_parent,
                p_nom=p_nom_converter,
                p_min_pu=-1.0,
                carrier='battery_link',
                efficiency=link_efficiency,
                marginal_cost=link_marginal_cost,
                p_nom_extendable=e_nom_extendable,
                capital_cost=annuitized_link_capex
            )

            n.add("Store", sto_i,
                  bus=store_internal_bus_name,
                  e_nom=e_nom_initial,
                  e_nom_extendable=e_nom_extendable,
                  e_cyclic=True,
                  e_max_pu=e_max_pu,
                  marginal_cost=marginal_cost_storage,
                  capital_cost=annuitized_e_capex,
                  lifetime=lifetime,
                  carrier=carrier)
            
            storage_count += 1
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Added Storage '{sto_i}' ({carrier}) at {bus_parent}, e_nom={e_nom_initial:.2f}, p_nom_converter={p_nom_converter:.2f}, extendable={e_nom_extendable}."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No storage data provided."


    # --------------------------
    # Constraints
    # --------------------------
    renewable_carriers = get_renewable_carriers()
    for carrier_name in renewable_carriers:
        if carrier_name in n.carriers.index:
            n.carriers.loc[carrier_name, 'renewable'] = 1.0
        else:
             yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Renewable carrier '{carrier_name}' not found in network carriers defined in this scenario."


    if co2_cap is not None and co2_cap > 0:
        n.add("GlobalConstraint", "CO2_CAP",
              carrier_attribute="co2_emissions",
              sense="<=",
              constant=float(co2_cap))
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Added CO2 emissions cap: {co2_cap:.2f} tons/year."

    if re_share is not None and re_share > 0:
        total_annual_demand = n.loads_t.p_set.sum().sum()
        if total_annual_demand == 0:
            yield f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Cannot apply RE share constraint as total annual demand is zero."
        else:
            min_re_generation = re_share * total_annual_demand
            n.add("GlobalConstraint", "RE_SHARE_MIN",
              carrier_attribute="renewable",
              sense=">=",
              constant=float(min_re_generation))
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Added RE share target: {re_share*100:.1f}% of total annual demand."


    # --------------------------
    # Optimization
    # --------------------------
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Starting optimization with solver: {solver}..."
    n.optimize(solver_name=solver)
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Optimization finished."
    
    # --- DIAGNOSTICS ---
    yield f"[{datetime.now().strftime('%H:%M:%S')}] --- OPTIMIZATION DIAGNOSTICS ---"
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Objective: {n.objective:.2f}" if n.objective is not None else "[Not Available]"
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Generators_t.p is empty: {n.generators_t.p.empty}"
    if not n.generators_t.p.empty:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Dispatch nonzero: {n.generators_t.p.sum().sum() > 0}"
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Dispatch shape: {n.generators_t.p.shape}"
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Loads_t.p_set is empty: {n.loads_t.p_set.empty}"
    if not n.loads_t.p_set.empty:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Loads nonzero: {n.loads_t.p_set.sum().sum() > 0}"
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Loads shape: {n.loads_t.p_set.shape}"
    yield f"[{datetime.now().strftime('%H:%M:%S')}] --- END DIAGNOSTICS ---"

    # --------------------------
    # Save outputs
    # --------------------------
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_folder_name = f"{scenario_name}_{current_datetime}"
    full_results_path_prefix = os.path.join(results_dir, run_folder_name)
    
    os.makedirs(full_results_path_prefix, exist_ok=True)

    n.export_to_netcdf(os.path.join(full_results_path_prefix, f"{scenario_name}.nc"))
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Network results saved to {os.path.join(full_results_path_prefix, f'{scenario_name}.nc')}"

    csv_sub_folder_path = os.path.join(full_results_path_prefix, "csv_outputs")
    os.makedirs(csv_sub_folder_path, exist_ok=True)
    n.export_to_csv_folder(csv_sub_folder_path) # Export all standard PyPSA CSVs first

    # --- NEW: Export generators_t.p (dispatch) to CSV ---
    if not n.generators_t.p.empty:
        dispatch_csv_path = os.path.join(csv_sub_folder_path, f"generators-dispatch_{scenario_name}.csv")
        n.generators_t.p.to_csv(dispatch_csv_path)
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Hourly dispatch data saved to {dispatch_csv_path}"
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No hourly dispatch data to save."
    # --- END NEW ---

    # --- NEW: Export summary_solar_wind_ls.csv ---
    try:
        solar_indices = n.generators[n.generators.carrier == 'Solar'].index
        wind_indices = n.generators[n.generators.carrier == 'Wind'].index
        
        if not solar_indices.empty or not wind_indices.empty:
            summary_data = []
            if not solar_indices.empty:
                gen_solar, cap_solar, cuf_solar, _, curt_solar, max_gen_solar, slack_cont_solar, peak_short_solar = calculate_metrics(n, 'Solar', solar_indices[0])
                summary_data.append({
                    'Generator_Type': 'Solar', 'Generation': gen_solar, 'Capacity': cap_solar, 'CUF': cuf_solar,
                    'Curtailment': curt_solar, 'Max_Generation': max_gen_solar,
                    'Slack_Contribution': slack_cont_solar, 'peak_shortage': peak_short_solar
                })
            if not wind_indices.empty:
                gen_wind, cap_wind, cuf_wind, _, curt_wind, max_gen_wind, slack_cont_wind, peak_short_wind = calculate_metrics(n, 'Wind', wind_indices[0])
                summary_data.append({
                    'Generator_Type': 'Wind', 'Generation': gen_wind, 'Capacity': cap_wind, 'CUF': cuf_wind,
                    'Curtailment': curt_wind, 'Max_Generation': max_gen_wind,
                    'Slack_Contribution': slack_cont_wind, 'peak_shortage': peak_short_wind
                })
            
            if summary_data:
                df_summary_metrics = pd.DataFrame(summary_data)
                summary_path = os.path.join(csv_sub_folder_path, f'summary_solar_wind_ls_{scenario_name}.csv')
                df_summary_metrics.to_csv(summary_path, index=False)
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Summary solar/wind metrics saved to {summary_path}"
        else:
            yield f"[{datetime.now().strftime('%H:%M:%S')}] No solar or wind generators to summarize metrics."
    except Exception as e:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Error exporting summary solar/wind metrics: {e}"
    # --- END NEW ---

    # --- CSV RENAMING LOGIC (for standard PyPSA CSVs) ---
    for filename in os.listdir(csv_sub_folder_path):
        if filename.endswith(".csv"):
            base_name = os.path.splitext(filename)[0]
            if not base_name.endswith(f"_{scenario_name}"):
                new_filename = f"{base_name}_{scenario_name}.csv"
                old_path = os.path.join(csv_sub_folder_path, filename)
                new_path = os.path.join(csv_sub_folder_path, new_filename)
                os.rename(old_path, new_path)
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Detailed CSV results saved and renamed with scenario tag to {csv_sub_folder_path}"
    # --- END CSV RENAMING LOGIC ---

    # --- NEW: Generate capacity_location.csv ---
    try:
        if not n.buses.empty and not n.generators.empty:
            buses_for_loc = n.buses[['x', 'y']].rename(columns={'x':'lon', 'y':'lat'})
            
            generators_for_loc = n.generators[['carrier', 'p_nom_opt', 'bus']].copy()
            grouped_generators_for_loc = generators_for_loc.groupby(['bus', 'carrier']).sum('p_nom_opt').reset_index()
            
            merged_capacity_location_df = pd.merge(buses_for_loc, grouped_generators_for_loc, left_index=True, right_on='bus', how='left')
            merged_capacity_location_df.rename(columns={'bus': 'Bus name'}, inplace=True)
            
            if 'lon_x' in merged_capacity_location_df.columns:
                merged_capacity_location_df['lon'] = merged_capacity_location_df['lon_x'].fillna(merged_capacity_location_df['lon_y'])
                merged_capacity_location_df['lat'] = merged_capacity_location_df['lat_x'].fillna(merged_capacity_location_df['lat_y'])
                merged_capacity_location_df.drop(columns=['lon_x', 'lat_x', 'lon_y', 'lat_y'], inplace=True)
            
            if 'bus' in merged_capacity_location_df.columns:
                 merged_capacity_location_df.drop(columns=['bus'], inplace=True)

            capacity_location_csv_path = os.path.join(full_results_path_prefix, f"capacity_location_{scenario_name}.csv")
            merged_capacity_location_df.to_csv(capacity_location_csv_path, index=False)
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Capacity location data saved to {capacity_location_csv_path}"
        else:
            yield f"[{datetime.now().strftime('%H:%M:%S')}] No bus or generator data to create capacity_location.csv."
    except Exception as e:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Error generating capacity_location.csv: {e}"
    # --- END NEW ---

    # --- NEW: Call create_plots for HTML outputs ---
    try:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Generating HTML plots..."
        create_plots(n, full_results_path_prefix, scenario_name, scenario_year)
        yield f"[{datetime.now().strftime('%H:%M:%S')}] HTML plots generated in plots/ subfolder."
    except Exception as e:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Error generating HTML plots: {e}"
    # --- END NEW ---

    yield f"[{datetime.now().strftime('%H:%M:%S')}] Scenario {scenario_name} finished. All results saved."
    
    yield (n, full_results_path_prefix)