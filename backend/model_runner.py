import pandas as pd
import pypsa
from datetime import datetime
import numpy as np
import os
import ast
import sys
import shutil

# --------------------------
# Helper functions
# --------------------------
def calculate_annuity(capital_cost, interest_rate, lifetime):
    """Convert CAPEX to annuitized annualized cost."""
    if lifetime <= 0:
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

def generate_report(n, inputs, results_path_prefix):
    """Generates a markdown report summarizing inputs and key optimization outputs."""
    report_file_path = os.path.join(results_path_prefix, "simulation_report.md")
    
    with open(report_file_path, "w") as f:
        f.write(f"# PacCEM Simulation Report: {inputs['scenario_name']}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## 1. Input Parameters\n")
        for key, value in inputs.items():
            if key in ['df_buses', 'df_generators', 'df_load', 'df_transmission_lines', 
                        'df_transformers', 'df_storage', 'df_generation_profiles', 
                        'df_scenario_year', 'enabled_techs', 'tech_cost_multipliers']:
                continue
            f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
        
        f.write("\n### Enabled Technologies:\n")
        for tech, enabled in inputs['enabled_techs'].items():
            f.write(f"- **{tech}:** {'Enabled' if enabled else 'Disabled'}\n")
        
        f.write("\n### Technology Cost Multipliers:\n")
        for tech, multiplier in inputs['tech_cost_multipliers'].items():
            f.write(f"- **{tech.title()}:** {multiplier:.2f}\n")

        f.write("\n---\n\n")
        f.write("## 2. Optimization Results Summary\n")
        
        if n.objective is not None:
            f.write(f"- **Total System Cost (Objective Value):** {n.objective:.2f} USD\n")
        else:
            f.write("- **Total System Cost (Objective Value):** Not available (optimization might not have run or failed)\n")

        # Total Generation by Carrier
        f.write("\n### Total Generation by Carrier (GWh/year)\n")
        if not n.generators_t.p.empty:
            total_generation = n.generators_t.p.sum().groupby(n.generators.carrier).sum() / 1e3
            for carrier, generation_gwh in total_generation.sort_values(ascending=False).items():
                f.write(f"- **{carrier.title()}:** {generation_gwh:.2f} GWh\n")
        else:
            f.write("- No generation data available.\n")

        # Installed Capacity (p_nom_opt)
        f.write("\n### Installed Generator Capacities (MW)\n")
        if not n.generators.empty:
            installed_capacities = n.generators.groupby('carrier').p_nom_opt.sum()
            for carrier, capacity_mw in installed_capacities.sort_values(ascending=False).items():
                f.write(f"- **{carrier.title()}:** {capacity_mw:.2f} MW\n")
        else:
            f.write("- No generator capacity data available.\n")

        # Storage Capacity (e_nom_opt)
        f.write("\n### Installed Storage Capacities (MWh)\n")
        if not n.stores.empty:
            installed_storage_capacities = n.stores.groupby('carrier').e_nom_opt.sum() if 'carrier' in n.stores.columns else n.stores.e_nom_opt.sum()
            if isinstance(installed_storage_capacities, pd.Series):
                for carrier, capacity_mwh in installed_storage_capacities.sort_values(ascending=False).items():
                    f.write(f"- **{carrier.title()}:** {capacity_mwh:.2f} MWh\n")
            else:
                f.write(f"- **Total Storage:** {installed_storage_capacities:.2f} MWh\n")
        else:
            f.write("- No storage capacity data available.\n")

        # Transmission Line Capacity (s_nom_opt)
        f.write("\n### Installed Transmission Line Capacities (MVA)\n")
        if not n.lines.empty:
            installed_line_capacities = n.lines.s_nom_opt.sum()
            f.write(f"- **Total Line Capacity:** {installed_line_capacities:.2f} MVA\n")
        else:
            f.write("- No transmission line capacity data available.\n")

        # Renewable Share Achieved
        if inputs['re_share'] is not None and not n.loads_t.p_set.empty and not n.generators_t.p.empty:
            total_annual_demand = n.loads_t.p_set.sum().sum()
            renewable_generation = n.generators_t.p.loc[:, n.generators.carrier.isin(get_renewable_carriers())].sum().sum()
            if total_annual_demand > 0:
                achieved_re_share = (renewable_generation / total_annual_demand) * 100
                f.write(f"\n- **Achieved Renewable Energy Share:** {achieved_re_share:.2f}% (Target: {inputs['re_share']*100:.1f}%)\n")
            else:
                f.write("\n- **Achieved Renewable Energy Share:** N/A (zero demand)\n")
        
        # Total CO2 Emissions
        if not n.generators_t.p.empty and 'co2_emissions' in n.carriers.columns:
            co2_emissions_per_carrier = n.generators_t.p.sum().groupby(n.generators.carrier).sum() * n.carriers.co2_emissions
            total_co2_emissions = co2_emissions_per_carrier.sum()
            f.write(f"\n- **Total CO2 Emissions:** {total_co2_emissions:.2f} tons/year (Cap: {inputs['co2_cap']} tons/year if set)\n")

        f.write("\n---\n\n")
        f.write("## 3. Detailed Results\n")
        f.write("Detailed results including time series data and component attributes are available in the `.nc` file and `.csv` files within the downloaded ZIP archive.\n")

    yield f"[{datetime.now().strftime('%H:%M:%S')}] Report generated at: {report_file_path}"
    return report_file_path


# --------------------------
# Comparison Tab Helper Functions (NEW/REFINED)
# --------------------------

def load_network_from_nc(file_path):
    """Loads a PyPSA network from a NetCDF file."""
    try:
        n = pypsa.Network(file_path)
        return n
    except Exception as e:
        raise ValueError(f"Failed to load network from {file_path}: {e}")

def get_time_series_component_info(n):
    """
    Returns a dictionary mapping PyPSA singular component names to their time-series dataframes,
    and a list of available attributes (columns) for each.
    This uses n.components_t which is the reliable way to get time-varying components.
    """
    info = {}
    
    component_type_map = {
        'generators_t': 'Generator',
        'loads_t': 'Load',
        'stores_t': 'Store',
        'buses_t': 'Bus',
        'lines_t': 'Line',
        'links_t': 'Link',
    }

    for df_attr_name in dir(n):
        if df_attr_name.endswith('_t') and hasattr(n, df_attr_name):
            ts_data_object = getattr(n, df_attr_name)
            
            if isinstance(ts_data_object, pd.DataFrame) and not ts_data_object.empty:
                singular_comp_type = component_type_map.get(df_attr_name, None)
                
                if singular_comp_type:
                    ts_attributes_to_check = {
                        'Generator': ['p', 'p_max_pu', 'p_min_pu'],
                        'Load': ['p_set', 'p'],
                        'Store': ['e', 'p'],
                        'Bus': ['marginal_price', 'p'],
                        'Line': ['p0', 'p1'],
                        'Link': ['p0', 'p1'],
                    }
                    
                    if singular_comp_type in ts_attributes_to_check:
                        for attr_name in ts_attributes_to_check[singular_comp_type]:
                            full_ts_attr = getattr(ts_data_object, attr_name, None)
                            
                            if isinstance(full_ts_attr, (pd.DataFrame, pd.Series)) and not full_ts_attr.empty:
                                info.setdefault(singular_comp_type, {'df_attr_prefix': df_attr_name, 'attributes': []})['attributes'].append(attr_name)
                                
    for comp_type in info:
        info[comp_type]['attributes'] = sorted(list(set(info[comp_type]['attributes'])))
        
    return info


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

def extract_selected_time_series(n, component_type, attribute, component_names):
    """
    Extracts time series data for specific components and attributes.
    Returns a DataFrame with index as snapshots and columns as component names.
    """
    df_ts = pd.DataFrame()
    
    comp_info = get_time_series_component_info(n)
    if component_type not in comp_info:
        return df_ts 
    
    df_attr_prefix = comp_info[component_type]['df_attr_prefix']
    
    if hasattr(getattr(n, df_attr_prefix), attribute):
        base_ts_data = getattr(getattr(n, df_attr_prefix), attribute)
        
        if isinstance(base_ts_data, pd.DataFrame) and not base_ts_data.empty:
            components_in_network = [comp for comp in component_names if comp in base_ts_data.columns]
            if components_in_network:
                df_ts = base_ts_data[components_in_network].copy()
                df_ts.index.name = 'Time'
        elif isinstance(base_ts_data, pd.Series) and not base_ts_data.empty:
            if not component_names or (base_ts_data.name and base_ts_data.name in component_names):
                df_ts = pd.DataFrame({base_ts_data.name if base_ts_data.name else attribute: base_ts_data}).copy()
                df_ts.index.name = 'Time'
            elif len(component_names) == 1 and not base_ts_data.name:
                df_ts = pd.DataFrame({component_names[0]: base_ts_data}).copy()
                df_ts.index.name = 'Time'
    return df_ts


# --------------------------
# Core model runner
# --------------------------
def run_model(
    project_name,
    results_dir,
    solver,
    co2_cap,
    re_share,
    slack_cost,
    discount_rate,
    demand_growth,
    tech_cost_multipliers,
    scenario_name,
    scenario_number,
    line_expansion,
    enabled_techs,
    default_new_gen_extendable,
    df_buses,
    df_generators,
    df_load,
    df_transmission_lines,
    df_transformers,
    df_storage,
    df_generation_profiles,
    df_scenario_year
):
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Starting simulation for scenario: {scenario_name}"
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Results will be saved to: {results_dir}"

    os.makedirs(results_dir, exist_ok=True)

    if solver.lower() == 'highs' and os.environ.get('PYPSA_SOLVER_HIGHSPY_PATH_SET') == 'true':
        yield f"[{datetime.now().strftime('%H:%M:%S')}] HiGHS executable path was successfully added to environment PATH by app startup."
    elif solver.lower() == 'highs':
         yield f"[{datetime.now().strftime('%H:%M:%S')}] Attempting to find HiGHS using system PATH (automatic detection in app.py failed)."


    # --------------------------
    # Scenario mapping
    # --------------------------
    target_year = 2023
    if not df_scenario_year.empty and 'Scenario' in df_scenario_year.columns and scenario_number in df_scenario_year['Scenario'].values:
        row = df_scenario_year[df_scenario_year['Scenario'] == scenario_number].iloc[0]
        target_year = int(row['Year'])
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Target Year for simulation: {target_year}"
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Demand Growth Factor: {demand_growth * 100:.1f}%"

    # --------------------------
    # Network build
    # --------------------------
    n = pypsa.Network()

    # Snapshots
    timestamps = pd.date_range(start=f'{target_year}-01-01 00:00', end=f'{target_year}-12-31 23:00', freq="h")[:8760]
    n.snapshots = timestamps
    n.snapshot_weightings['objective'] = n.snapshot_weightings['objective'] * (8760 / len(timestamps))
    yield f"[{datetime.now().strftime('%H:%M:%S')}] PyPSA Network initialized with {len(timestamps)} snapshots."

    # Buses
    if not df_buses.empty:
        if 'Bus name' in df_buses.columns:
            df_buses_processed = df_buses.set_index('Bus name', drop=False).copy()
        else:
            df_buses_processed = df_buses.copy()
            df_buses_processed.index.name = 'Bus name'

        for bus_name, row in df_buses_processed.iterrows():
            n.add("Bus",
                bus_name,
                x=row.get("x"),
                y=row.get("y"))
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Added {len(df_buses_processed)} buses."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No bus data provided."
        raise ValueError("No bus data provided to build the network.")


    # Carriers
    all_carriers_from_generators = df_generators['Carrier'].dropna().astype(str).unique()
    all_carriers_from_storage = df_storage['Carrier'].dropna().astype(str).unique()

    all_carriers_from_data = set(all_carriers_from_generators).union(set(all_carriers_from_storage))
    
    standard_carriers = {"electricity", "backup", "AC", "storage_charge", "storage_discharge", "slack"}
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

    # Demand
    if not df_load.empty:
        df_load_processed = df_load.copy() 

        if not isinstance(df_load_processed.index, pd.DatetimeIndex):
            if len(df_load_processed) == len(n.snapshots):
                df_load_processed.index = n.snapshots
            else:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Load data rows ({len(df_load_processed)}) do not match snapshot count ({len(n.snapshots)}). Attempting to reindex, but data might be misaligned."
                df_load_processed = df_load_processed.reindex(n.snapshots)

        df_load_processed = df_load_processed.reindex(n.snapshots).fillna(0)

        numeric_cols = df_load_processed.select_dtypes(include=np.number).columns
        df_load_processed = df_load_processed[numeric_cols]
        df_load_processed.dropna(axis=1, how='all', inplace=True)

        if df_load_processed.empty:
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Processed load data is empty or non-numeric. Skipping load addition."
        else:
            for load_centre in df_load_processed.columns:
                if load_centre in n.buses.index:
                    load_fix = pd.Series(df_load_processed[load_centre] * (1 + demand_growth), index=n.snapshots, name=load_centre)
                    n.add("Load", load_centre, bus=load_centre, p_set=load_fix)
                else:
                    yield f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Load center '{load_centre}' does not have a corresponding bus. Skipping."
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Added {len(n.loads)} load components."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No load data provided."


    # Process Generation Profiles
    W_gen_profile = pd.Series(np.ones(len(n.snapshots)), index=n.snapshots)
    S_gen_profile = pd.Series(np.ones(len(n.snapshots)), index=n.snapshots)
    H_gen_profile = pd.Series(np.ones(len(n.snapshots)), index=n.snapshots)

    if not df_generation_profiles.empty:
        if not isinstance(df_generation_profiles.index, pd.DatetimeIndex):
            if len(df_generation_profiles) == len(n.snapshots):
                df_gen_profiles_processed = df_generation_profiles.set_index(n.snapshots).copy()
            else:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Generation profile rows ({len(df_generation_profiles)}) do not match snapshot count ({len(n.snapshots)}). Profiles might be misaligned."
                df_gen_profiles_processed = df_generation_profiles.reindex(n.snapshots).fillna(0)
        else:
            df_gen_profiles_processed = df_generation_profiles.reindex(n.snapshots).fillna(0).copy()

        if enabled_techs.get('Wind', False) and 'Wind profile' in df_gen_profiles_processed.columns:
            W_gen_profile = pd.to_numeric(df_gen_profiles_processed['Wind profile'], errors='coerce').fillna(1.0)
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Wind generation profile loaded."
        if (enabled_techs.get('Solar', False) or enabled_techs.get('Solar Rooftop', False)) and 'Solar profile' in df_gen_profiles_processed.columns:
            S_gen_profile = pd.to_numeric(df_gen_profiles_processed['Solar profile'], errors='coerce').fillna(1.0)
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Solar generation profile loaded."
        if enabled_techs.get('Hydro', False) and 'Hydro profile' in df_gen_profiles_processed.columns:
            H_gen_profile = pd.to_numeric(df_gen_profiles_processed['Hydro profile'], errors='coerce').fillna(1.0)
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Hydro generation profile loaded."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No generation profile data provided. Intermittent generators will use default p_max_pu=1 (constant)."


    # --------------------------
    # Generators
    # --------------------------
    if not df_generators.empty:
        if 'Generator name' in df_generators.columns:
            df_generators_processed = df_generators.set_index('Generator name', drop=True).copy()
        else:
            df_generators_processed = df_generators.copy()
            df_generators_processed.index.name = 'Generator name'
        
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
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Generator '{gen_i}' refers to non-existent bus '{bus}'. Skipping."
                continue

            status = row.get('Status', 0) # 0 for existing, 1 for new
            p_nom_initial = float(row.get('Capacity(MW)', 0.0)) if status == 0 else 0.0

            input_p_nom_extendable_raw = row.get('p_nom_extendable')
            
            if input_p_nom_extendable_raw is not None and pd.notna(input_p_nom_extendable_raw):
                p_nom_extendable = bool(input_p_nom_extendable_raw)
            elif status == 1:
                p_nom_extendable = default_new_gen_extendable
            else:
                p_nom_extendable = False

            lifetime = int(row.get('lifetime', 25))
            marginal_cost = float(row.get('Variable cost (USD/MWh)', 0.0))
            raw_capex = float(row.get('Capital_cost (USD/MW)', 0.0))
            min_generation_level = float(row.get('min_generation_level', 0.0))
            efficiency = float(row.get('efficiency', 1.0))
            committable = bool(row.get('committable', False))
            
            scaled_capex = apply_cost_multiplier(carrier, raw_capex, tech_cost_multipliers)
            annuitized_capex = calculate_annuity(scaled_capex, discount_rate, lifetime)

            p_max_pu_value = 1.0
            if carrier == 'Wind':
                p_max_pu_value = W_gen_profile
            elif carrier in ('Solar', 'Solar Rooftop'):
                p_max_pu_value = S_gen_profile
            elif carrier == 'Hydro':
                p_max_pu_value = H_gen_profile

            qty = int(row.get('Quantity', 1))
            size_mw = float(row.get('Size (MW)', p_nom_initial))

            if status == 0 and qty > 1 and pd.notna(size_mw) and size_mw > 0:
                for i in range(1, qty + 1):
                    gen_name_instance = f"{gen_i}_{carrier[:2].upper()}{i}"
                    if gen_name_instance in n.generators.index:
                        gen_name_instance = f"{gen_i}_{carrier[:2].upper()}{i}_{added_generators_count}"
                    n.add("Generator",
                        gen_name_instance,
                        bus=bus,
                        p_nom=size_mw,
                        p_min_pu=min_generation_level,
                        p_max_pu=p_max_pu_value,
                        carrier=carrier,
                        efficiency=efficiency,
                        marginal_cost=marginal_cost,
                        capital_cost=annuitized_capex,
                        p_nom_extendable=p_nom_extendable,
                        committable=committable,
                        ramp_limit_down=1, ramp_limit_up=1,
                        p_nom_initial=p_nom_initial 
                    )
                    added_generators_count += 1
                    yield f"[{datetime.now().strftime('%H:%M:%S')}] Added Existing Gen '{gen_name_instance}' ({carrier}) at {bus}, p_nom={size_mw}, marginal_cost={marginal_cost} USD/MWh."
            else:
                n.add("Generator",
                    gen_i,
                    bus=bus,
                    p_nom=p_nom_initial,
                    p_min_pu=min_generation_level,
                    p_max_pu=p_max_pu_value,
                    carrier=carrier,
                    efficiency=efficiency,
                    marginal_cost=marginal_cost,
                    capital_cost=annuitized_capex,
                    p_nom_extendable=p_nom_extendable,
                    committable=committable,
                    ramp_limit_down=1, ramp_limit_up=1,
                    p_nom_initial=p_nom_initial
                )
                added_generators_count += 1
                status_str = "New" if status == 1 else "Existing" 
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Added {status_str} Gen '{gen_i}' ({carrier}) at {bus}, p_nom={p_nom_initial}, marginal_cost={marginal_cost} USD/MWh, extendable={p_nom_extendable}."

        slack_bus = list(n.buses.index)[0] if not n.buses.empty else "DummyBusForSlack"
        if slack_bus not in n.buses.index:
            n.add("Bus", slack_bus)

        n.add("Generator",
                'slack',
                bus=slack_bus,
                p_nom=1e6,
                p_max_pu=1,
                marginal_cost=slack_cost,
                carrier='slack')
        added_generators_count += 1
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Added slack generator at bus '{slack_bus}' with marginal_cost={slack_cost} USD/MWh."
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Total {added_generators_count} generators added."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No generator data provided."

    # --------------------------
    # Transmission Lines
    # --------------------------
    if not df_transmission_lines.empty:
        for i, row in df_transmission_lines.iterrows():
            from_bus = row['From']
            to_bus = row['To']
            if from_bus not in n.buses.index or to_bus not in n.buses.index:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Transmission line between '{from_bus}' and '{to_bus}' refers to non-existent bus(es). Skipping."
                continue

            name = f"Line_{from_bus}_{to_bus}_{i}"
            s_nom_extendable = bool(row.get('s_nom_extendable', line_expansion))
            
            line_lifetime = int(row.get('lifetime', 25))
            line_capital_cost_raw = float(row.get('Capital_cost (USD/MW/km)', 0.0))
            annuitized_line_capex = calculate_annuity(line_capital_cost_raw, discount_rate, line_lifetime)

            n.add("Line",
                  name,
                  bus0=from_bus,
                  bus1=to_bus,
                  type=row.get('type', None),
                  s_nom=float(row.get('s_nom', 0.0)) if not s_nom_extendable else 0.0,
                  s_nom_extendable=s_nom_extendable,
                  capital_cost=annuitized_line_capex,
                  length=float(row.get('Length (kM)', 1.0)))
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Added Line '{name}': {from_bus} <-> {to_bus}, s_nom={row.get('s_nom',0.0)}, extendable={s_nom_extendable}."
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Added {len(n.lines)} transmission lines."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No transmission line data provided."

    # --------------------------
    # Transformers
    # --------------------------
    if not df_transformers.empty:
        for i, row in df_transformers.iterrows():
            bus0 = str(row['bus0']).strip()
            bus1 = str(row['bus1']).strip()
            if bus0 not in n.buses.index or bus1 not in n.buses.index:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Transformer '{row.get('Location', f'Trans_{i}')}' between '{bus0}' and '{bus1}' refers to non-existent bus(es). Skipping."
                continue
            
            name = str(row.get('Location', f"Transformer_{i}")).strip()

            transformer_lifetime = int(row.get('lifetime', 25))
            transformer_capital_cost_raw = float(row.get('Capital_cost (USD)', 0.0))
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
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Added Transformer '{name}': {bus0} <-> {bus1}, s_nom={row.get('s_nom', 0.0)}."
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Added {len(n.transformers)} transformers."
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No transformer data provided."


    # --------------------------
    # Storage construction
    # --------------------------
    if not df_storage.empty:
        if 'name' in df_storage.columns:
            df_storage_processed = df_storage.set_index('name', drop=True).copy()
        else:
            df_storage_processed = df_storage.copy()
            df_storage_processed.index.name = 'name'

        if 'Scenario' in df_storage_processed.columns:
             df_storage_processed['Scenario'] = df_storage_processed['Scenario'].apply(
                 lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x
             )

        storage_count = 0
        for sto_i, row in df_storage_processed.iterrows():
            bus = row['Bus']
            if bus not in n.buses.index:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Storage '{sto_i}' refers to non-existent bus '{bus}'. Skipping."
                continue
            
            if 'Scenario' in row.index and not (scenario_number in row['Scenario'] if isinstance(row['Scenario'], list) else False):
                 yield f"[{datetime.now().strftime('%H:%M:%S')}] Skipping storage '{sto_i}' as it's not part of the current scenario."
                 continue
            
            carrier = row.get('Carrier', 'BESS')
            if not enabled_techs.get(carrier, True):
                yield f"[{datetime.now().strftime('%H:%M:%S')}] Skipping storage '{sto_i}' (Carrier: {carrier}) as its technology is disabled."
                continue

            st_bus_name = f"{bus}_st_sec_{sto_i}"
            if st_bus_name not in n.buses.index:
                n.add("Bus", st_bus_name)
            
            status = row.get('Status', 0)
            e_nom = float(row.get('Capacity(MW)', 0.0)) if status == 0 else 0.0
            e_nom_extendable = bool(row.get('e_nom_extendable', False))
            lifetime = int(row.get('lifetime', 20))
            marginal_cost = float(row.get('Variable cost (USD/MWh)', 0.0))
            raw_capex_per_mwh = float(row.get('Capital_cost (USD/MWh)', 0.0))

            annuitized_e_capex = calculate_annuity(raw_capex_per_mwh, discount_rate, lifetime)

            charge_efficiency = float(row.get('charge_efficiency', 0.90))
            discharge_efficiency = float(row.get('discharge_efficiency', 1.0))

            p_nom_converter = float(row.get('p_nom_converter', 100.0))
            link_capital_cost_raw = float(row.get('link_capital_cost_per_mw', 0.0))
            annuitized_link_capex = calculate_annuity(link_capital_cost_raw, discount_rate, lifetime)

            n.add(
                "Link", f"{sto_i}_charge",
                bus1=st_bus_name,
                bus0=bus,
                p_nom=p_nom_converter,
                carrier='storage_charge',
                efficiency=charge_efficiency,
                p_nom_extendable=e_nom_extendable,
                capital_cost=annuitized_link_capex
            )
            n.add(
                "Link", f"{sto_i}_discharge",
                bus0=st_bus_name,
                bus1=bus,
                p_nom=p_nom_converter,
                carrier='storage_discharge',
                efficiency=discharge_efficiency,
                p_nom_extendable=e_nom_extendable,
                capital_cost=annuitized_link_capex
            )
            
            n.add("Store", sto_i,
                  bus=st_bus_name,
                  e_nom=e_nom,
                  marginal_cost=marginal_cost,
                  e_cyclic=True,
                  e_max_pu=float(row.get('e_max_pu', 0.9)),
                  e_nom_extendable=e_nom_extendable,
                  capital_cost=annuitized_e_capex)
            
            storage_count += 1
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Added Storage '{sto_i}' ({carrier}) at {bus}, e_nom={e_nom}, extendable={e_nom_extendable}."
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Total {storage_count} storage units added."
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
             yield f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Renewable carrier '{carrier_name}' not found in network carriers defined in this scenario."


    if co2_cap is not None and co2_cap > 0:
        n.add("GlobalConstraint", "CO2_CAP",
              carrier_attribute="co2_emissions",
              sense="<=",
              constant=float(co2_cap))
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Added CO2 emissions cap: {co2_cap} tons/year."

    if re_share is not None and re_share > 0:
        total_annual_demand = n.loads_t.p_set.sum().sum()
        if total_annual_demand == 0:
            yield f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Cannot apply RE share constraint as total annual demand is zero."
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
    n.export_to_csv_folder(csv_sub_folder_path)

    # --- NEW: Export generators_t.p (dispatch) to CSV ---
    if not n.generators_t.p.empty:
        dispatch_csv_path = os.path.join(csv_sub_folder_path, f"generators-dispatch_{scenario_name}.csv")
        n.generators_t.p.to_csv(dispatch_csv_path)
        yield f"[{datetime.now().strftime('%H:%M:%S')}] Hourly dispatch data saved to {dispatch_csv_path}"
    else:
        yield f"[{datetime.now().strftime('%H:%M:%S')}] No hourly dispatch data to save."
    # --- END NEW ---

    for filename in os.listdir(csv_sub_folder_path):
        if filename.endswith(".csv"):
            base_name = os.path.splitext(filename)[0]
            # Avoid renaming the already custom-named dispatch file if it exists
            if not base_name.startswith("generators-dispatch_"): 
                new_filename = f"{base_name}_{scenario_name}.csv"
                os.rename(os.path.join(csv_sub_folder_path, filename), os.path.join(csv_sub_folder_path, new_filename))
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Detailed CSV results saved and renamed with scenario tag to {csv_sub_folder_path}"

    inputs_for_report = {
        'project_name': project_name, 'results_dir': results_dir, 'solver': solver,
        'co2_cap': co2_cap, 're_share': re_share, 'slack_cost': slack_cost,
        'discount_rate': discount_rate, 'demand_growth': demand_growth,
        'tech_cost_multipliers': tech_cost_multipliers, 'scenario_name': scenario_name,
        'scenario_number': scenario_number, 'line_expansion': line_expansion,
        'enabled_techs': enabled_techs,
        'default_new_gen_extendable': default_new_gen_extendable
    }
    report_path = yield from generate_report(n, inputs_for_report, full_results_path_prefix)

    yield f"[{datetime.now().strftime('%H:%M:%S')}] Scenario {scenario_name} finished. All results saved."
    
    yield (n, full_results_path_prefix)