import pandas as pd
import numpy as np
import os
import xarray as xr
import networkx as nx

from src.configManager import ConfigManager
from src.eaglei_modules import constants
from src.eaglei_modules.eagleiEventProcessing import apply_spatiotemporal_grouping_optimized, apply_spatiotemporal_grouping
# from src.eaglei_modules.eagleiEventProcessing import apply_spatiotemporal_grouping_optimized2

def create_multi_county_events(state_county_pairs: list[dict], 
                               start_year: int, 
                               end_year: int, 
                               config: ConfigManager) -> None:
    """
    Create multi-county outage events by merging outage data from multiple counties
    and applying time+location based event detection across the combined dataset.
    
    Parameters:
    -----------
    state_county_pairs : list of dict
        List of dictionaries with 'state' and 'county' keys
    start_year : int
        Start year for analysis
    end_year : int
        End year for analysis
    config : ConfigManager
        Configuration manager object
    """
    
    datasets = {}
    df_fips = pd.DataFrame()
    for pair in state_county_pairs:
        state = pair['state']
        county = pair['county']
        print(f"Processing county: {county}, {state} ({start_year}-{end_year})")
        # Load the merged outage and weather netcdf file for the county
        merged_file_dir = os.path.join(config.get("data_paths.merged_data_dir"), state)
        merged_file_name = config.get("file_patterns.merged_file_pattern").format(start=start_year, end=end_year, county=county, state=state)
        merged_file_path = os.path.join(merged_file_dir, merged_file_name)
        
        if not os.path.exists(merged_file_path):
            raise FileNotFoundError(f"Merged NetCDF file not found: {merged_file_path}")
        
        # Open the county dataset
        ds_county = xr.open_dataset(merged_file_path)
        county_name = state + county
        datasets[county_name] = ds_county
        
        # After successfully processing the county, use the attributes from ds_county to build the fips dataframe
        df_fips = pd.concat([df_fips, pd.DataFrame({'state': [state],
                                                    'state_fips_code': [constants.STATE_FIPS_DICT.get(state.lower(), None)],
                                                    'county': [county], 
                                                    'county_fips_code': [ds_county.attrs.get('county_fips_code', None)]})], ignore_index=True)
        # Close the individual dataset
        ds_county.close()

    # Weather station data with dimensions (time, station)
    datasets_renamed = {}
    for county_name, ds in datasets.items():
        # Extract only weather variables
        weather_vars = [v for v in ds.data_vars if v not in ['customers_out', 'event_number_eaglei']]
        ds_copy = ds[weather_vars].copy()
        ds_copy['station'] = [f"{county_name}_{stn}" for stn in ds['station'].values]
        datasets_renamed[county_name] = ds_copy

    weather_data = xr.concat(datasets_renamed.values(), dim='station')

    # Add county as a coordinate for each station
    # county_labels = []
    # for county_name, ds in datasets_renamed.items():
    #     county_labels.extend([county_name] * len(ds['station']))
    # weather_data = weather_data.assign_coords(station_county=('station', county_labels))

    # Outage data with dimensions (time, county)
    customers_out_dict = {}
    events_id_dict = {}
    for county_name, ds in datasets.items():
        customers_out_dict[county_name] = ds['customers_out']
        events_id_dict[county_name] = ds['event_number_eaglei']

    outage_data = xr.Dataset({
        'customers_out': xr.concat(customers_out_dict.values(), dim='county'),
        'event_number_eaglei': xr.concat(events_id_dict.values(), dim='county')
    })
    outage_data['county'] = list(customers_out_dict.keys())

    # Store both in a single structure
    ds_combined = weather_data.copy()
    # add outage variables with (time, county) dimensions
    ds_combined = ds_combined.assign(
        customers_out=outage_data['customers_out'],
        event_number_eaglei=outage_data['event_number_eaglei']
    )


    if ds_combined is not None:
        # Apply multi-county event detection on the combined dataset
        print("Applying multi-county event detection...")
        
        # 1. Create adjacency graph for the counties
        from src.eaglei_modules.eagleiEventProcessing import create_multi_state_county_adjacency_graph, analyze_county_graph, visualize_county_graph
        adjacency_graph = create_multi_state_county_adjacency_graph(df_fips['county_fips_code'].to_list())

        # Check if there are any custom adjacency changes specified in the config
        adjacency_changes = config.get("county_adjacency_changes", {})
        if adjacency_changes:
            # Apply additions
            for edge in adjacency_changes.get("add_edges", []):
                from_state = edge['from_state'].replace(" ", "").title()
                from_county = edge['from_county'].replace(" ", "").title()
                to_state = edge['to_state'].replace(" ", "").title()
                to_county = edge['to_county'].replace(" ", "").title()

                from_node_name = from_state + from_county
                to_node_name = to_state + to_county

                # check if both nodes exist in the graph
                if adjacency_graph.has_node(from_node_name) and adjacency_graph.has_node(to_node_name):
                    # check if edge already exists
                    if not adjacency_graph.has_edge(from_node_name, to_node_name):
                        adjacency_graph.add_edge(from_node_name, to_node_name, weight=0, overlap_length_km=0, custom_added=True)
                        print(f"Added custom edge between {from_county}, {from_state} and {to_county}, {to_state}")
            
            # Apply deletions
            for edge in adjacency_changes.get("del_edges", []):
                from_state = edge['from_state'].replace(" ", "").title()
                from_county = edge['from_county'].replace(" ", "").title()
                to_state = edge['to_state'].replace(" ", "").title()
                to_county = edge['to_county'].replace(" ", "").title()
                
                from_node_name = from_state + from_county
                to_node_name = to_state + to_county

                # check if both nodes exist in the graph
                if adjacency_graph.has_node(from_node_name) and adjacency_graph.has_node(to_node_name):
                    # check if edge exists
                    if adjacency_graph.has_edge(from_node_name, to_node_name):
                        adjacency_graph.remove_edge(from_node_name, to_node_name)
                        print(f"Removed custom edge between {from_county}, {from_state} and {to_county}, {to_state}")
        
        analyze_county_graph(adjacency_graph)

        visualize_county_graph(adjacency_graph, save=True, save_path=os.path.join(config.get("data_paths.results_dir"), "county_adjacency_graph.png"))

        # 2. Assign unique event ids to county events to avoid conflicts
        print("Assigning unique event IDs to county events...")
        ds_combined = _assign_unique_event_ids(ds_combined)

        # 3. Identify temporally overlapping outages across neighboring counties and merge them into single events
        print("Applying temporal event grouping across all counties...")
        ds_combined = _apply_temporal_event_grouping(ds_combined, new_event_column_name='event_number_mc_temporal')

        # 4. Apply spatiotemporal event grouping within each temporal event
        print("Applying spatiotemporal event grouping within each temporal event...")
        ds_combined = _apply_spatiotemporal_event_grouping(
            ds_combined,
            county_adjacency_graph=adjacency_graph,
            county_event_col_name='event_number_eaglei',
            temporal_event_col_name='event_number_mc_temporal',
            new_event_column_name='event_number_mc_spatiotemporal'
        )
        
        # Remove existing attributes if present
        if 'state' in ds_combined.attrs:
            del ds_combined.attrs['state']
        if 'county' in ds_combined.attrs:
            del ds_combined.attrs['county']
        if 'county_fips_code' in ds_combined.attrs:
            del ds_combined.attrs['county_fips_code']
        if 'total_county_customers' in ds_combined.attrs:
            del ds_combined.attrs['total_county_customers']
        if 'event_customer_threshold' in ds_combined.attrs:
            del ds_combined.attrs['event_customer_threshold']
        if 'event_threshold_method' in ds_combined.attrs:
            del ds_combined.attrs['event_threshold_method']

        # Find number of events detected
        unique_events = ds_combined['event_number_eaglei'].values
        n_events_county = len(np.unique(unique_events[unique_events>0]).astype(int))
        unique_events = ds_combined['event_number_mc_temporal'].values
        n_events_temporal = len(np.unique(unique_events[unique_events>0]).astype(int))
        unique_events = ds_combined['event_number_mc_spatiotemporal'].values
        n_events_spatiotemporal = len(np.unique(unique_events[unique_events>0]).astype(int))

        # Update and add attributes to reflect multi-county data
        ds_combined.attrs.update({
            'title': f'Merged Outage and Weather Data for {len(df_fips)} counties, ({start_year}-{end_year})',
            'description': f'This dataset contains combined and cleaned outage and weather data from {len(df_fips)} counties from {start_year} to {end_year}.',
            'start_year': start_year,
            'end_year': end_year,
            'states': ', '.join(sorted(df_fips['state'].unique())),
            'state_fips_codes': ', '.join(df_fips['state_fips_code'].astype(str).unique().tolist()),
            'counties': ', '.join(sorted(df_fips['county'].unique())),
            'county_fips_codes': ', '.join(df_fips['county_fips_code'].astype(str).unique().tolist()),
            'temporal_resolution': '15 minutes',
            'total_county_events': n_events_county,
            'total_temporal_events': n_events_temporal,
            'total_spatiotemporal_events': n_events_spatiotemporal,
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Save the resulting events dataset
        multi_county_data_file_dir = config.get("data_paths.multi_county_data_dir")
        os.makedirs(multi_county_data_file_dir, exist_ok=True)
        multi_county_data_file_name = config.get("file_patterns.multi_county_data_file_pattern").format(start=start_year, end=end_year, nCounties=len(df_fips))
        multi_county_data_file_path = os.path.join(multi_county_data_file_dir, multi_county_data_file_name)
        
        # Check if file already exists
        if os.path.exists(multi_county_data_file_path):
            print(f"Warning: Overwriting existing multi-county dataset at {multi_county_data_file_path}")
        
        # Save the combined dataset
        print(f"Saving multi-county dataset to {multi_county_data_file_path}")
        try:
            ds_combined.to_netcdf(multi_county_data_file_path)
        except PermissionError as pe:
            print(f"Permission error while saving multi-county dataset: {pe}")
        except Exception as e:
            print(f"Error saving multi-county dataset: {e}")
        
        # Close the combined dataset
        ds_combined.close()
        
        print("Multi-county events processing completed.")
    else:
        print("No county datasets were combined. Exiting.")
    
    return None


def _assign_unique_event_ids(dataset: xr.Dataset) -> xr.Dataset:
    """
    Assign unique event IDs to county events in the dataset to avoid conflicts.
    Parameters:
    -----------
    dataset : xr.Dataset
        Xarray dataset containing county-level event numbers in 'event_number_eaglei' variable.
    Returns:
    --------
    xr.Dataset
        Updated xarray dataset with unique event IDs assigned.
    """

    # Extract the event numbers (shape: county, time)
    event_numbers = dataset['event_number_eaglei'].values.copy()

    # Create a mapping for unique event IDs
    next_unique_id = 1
    county_event_mapping = {}  # key: (county_idx, original_event_num), value: unique_event_num

    # New array for unique event numbers
    unique_event_numbers = np.zeros_like(event_numbers)

    n_counties, n_times = event_numbers.shape

    for c in range(n_counties):
        for t in range(n_times):
            original_event = event_numbers[c, t]
            
            # Preserve special values
            if original_event == 0 or original_event == -1:
                unique_event_numbers[c, t] = original_event
                continue
            
            # Check if this county-event combination has been seen
            key = (c, original_event)
            if key not in county_event_mapping:
                # Assign new unique ID
                county_event_mapping[key] = next_unique_id
                next_unique_id += 1
            
            unique_event_numbers[c, t] = county_event_mapping[key]

    # Update the dataset with unique event numbers
    dataset['event_number_eaglei'] = (('county', 'time'), unique_event_numbers)

    # Verify the result
    print(f"Total unique events (excluding 0 and -1): {len(county_event_mapping)}")
    print(f"Event numbers range from: {unique_event_numbers[unique_event_numbers > 0].min()} to {unique_event_numbers[unique_event_numbers > 0].max()}")
    print(f"Count of 0 values: {(unique_event_numbers == 0).sum()}")
    print(f"Count of -1 values: {(unique_event_numbers == -1).sum()}")

    # Show some summary for the counties
    for county_idx in range(n_counties):
        county_name = dataset['county'].values[county_idx]
        county_events = np.unique(unique_event_numbers[county_idx])
        county_events = county_events[county_events > 0]  # Exclude 0 and -1
        print(f"{county_name}: {len(county_events)} unique events, IDs range: {county_events.min() if len(county_events) > 0 else 'N/A'} to {county_events.max() if len(county_events) > 0 else 'N/A'}")
    
    return dataset


def _apply_temporal_event_grouping(dataset: xr.Dataset, new_event_column_name: str = 'event_number_mc_temporal') -> xr.Dataset:
    """
    Apply temporal event grouping across all counties to identify overlapping events in time.
    Parameters:
    -----------
    dataset : xr.Dataset
        Xarray dataset containing county-level event numbers in 'event_number_eaglei' variable.
    new_event_column_name : str
        Name of the new variable to store temporal event grouping results.
    Returns:
    --------
    xr.Dataset
        Updated xarray dataset with temporal event grouping results.
    """

    # Convert relevant data to pandas DataFrame for processing
    temp = dataset[['time','county','customers_out','event_number_eaglei']].to_dataframe().reset_index()
    temp = temp[temp['event_number_eaglei']>0].reset_index(drop=True)

    from src.eaglei_modules.eagleiEventProcessing import find_time_overlapping_groups
    temp = find_time_overlapping_groups(temp, 
                                        event_col='event_number_eaglei',
                                        new_event_col=new_event_column_name,
                                        timestamp_col='time',
                                        county_col='county')
    
    # Create a new array initialized with 0 (for non-event times)
    n_counties = len(dataset['county'])
    n_times = len(dataset['time'])
    
    new_event_array = np.zeros((n_counties, n_times), dtype=int)
    
    # Create lookup dictionaries for fast indexing
    county_to_idx = {county: idx for idx, county in enumerate(dataset['county'].values)}
    time_to_idx = {pd.Timestamp(time): idx for idx, time in enumerate(dataset['time'].values)}
    
    # Fill in the new event numbers
    for _, row in temp.iterrows():
        county_idx = county_to_idx[row['county']]
        time_idx = time_to_idx[pd.Timestamp(row['time'])]
        new_event_array[county_idx, time_idx] = row[new_event_column_name]
    
    # Add to dataset
    dataset[new_event_column_name] = (('county', 'time'), new_event_array)
    
    print(f"\nAdded '{new_event_column_name}' to dataset with temporal event grouping.")
    print(f"Unique temporal events: {len(np.unique(new_event_array[new_event_array > 0]))}")
    
    return dataset


def _apply_spatiotemporal_event_grouping(dataset: xr.Dataset, 
                                         county_adjacency_graph: nx.Graph,
                                         county_event_col_name: str = 'event_number_eaglei',
                                         temporal_event_col_name: str = 'event_number_mc_temporal', 
                                         new_event_column_name: str = 'event_number_mc_spatiotemporal') -> xr.Dataset:
    '''
    Apply spatiotemporal event grouping within each temporal event to identify events that are both temporally overlapping and spatially adjacent.
    Parameters:
    -----------
    dataset : xr.Dataset
        Xarray dataset containing county-level event numbers in 'event_number_eaglei' variable and temporal event numbers in 'event_number_mc_temporal'.
    county_adjacency_graph : nx.Graph
        NetworkX graph representing adjacency relationships between counties.
    county_event_col_name : str
        Name of the variable containing county-level event numbers.
    temporal_event_col_name : str
        Name of the variable containing temporal event numbers.
    new_event_column_name : str
        Name of the new variable to store spatiotemporal event grouping results.
    Returns:
    --------
    xr.Dataset
        Updated xarray dataset with spatiotemporal event grouping results.
    '''

    # dataset, df_result = apply_spatiotemporal_grouping(
    #     combined=dataset,
    #     graph_of_counties=county_adjacency_graph,
    #     temporal_event_col=temporal_event_col_name,
    #     county_event_col=county_event_col_name,
    #     neighbour_level=1,
    #     time_overlap_method='outage_process_overlap',
    #     verbose=1
    # )

    dataset, df_result = apply_spatiotemporal_grouping_optimized(
        combined=dataset,
        graph_of_counties=county_adjacency_graph,
        temporal_event_col=temporal_event_col_name,
        county_event_col=county_event_col_name,
        neighbour_level=1,
        time_overlap_method='outage_process_overlap',
        verbose=1
    )

    # dataset, df_result = apply_spatiotemporal_grouping_optimized2(
    #     combined=dataset,
    #     graph_of_counties=county_adjacency_graph,
    #     temporal_event_col=temporal_event_col_name,
    #     county_event_col=county_event_col_name,
    #     neighbour_level=1,
    #     time_overlap_method='outage_process_overlap',
    #     verbose=1
    # )

    return dataset