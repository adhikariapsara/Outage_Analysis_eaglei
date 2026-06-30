import read_files
import func

groups=[
    ["Washington", "Washington_rucc1", True, "#2166AC", 0, 2015, 2025],
    ["Washington", "Washington_nonrucc1", True, "#D1E5F0", 0, 2015, 2025],
    ["Illinois", "Illinois_rucc1", True, "#54278F", 0, 2015, 2025],
    ["Illinois", "Illinois_nonrucc1", True, "#DADAEB", 0, 2015, 2025],
    ["New York", "New York_rucc1", True, "#8C510A", 0, 2015, 2025],
    ["New York", "New York_nonrucc1", True, "#DFC27D", 0, 2015, 2025],
    ["Massachusetts", "Massachusetts_rucc1", True, "#1B7837", 0, 2015, 2025],
    ["Massachusetts", "Massachusetts_nonrucc1", True, "#D9F0D3", 0, 2015, 2025],
    ["California", "California_rucc1", True, "#B2182B", 0, 2015, 2025],
    ["California", "California_nonrucc1", True, "#F4CCCC", 0, 2015, 2025],
    ["Arizona", "Arizona_rucc1", True, "#C51B7D", 0, 2015, 2025],
    ["Arizona", "Arizona_nonrucc1", True, "#F4CAE4", 0, 2015, 2025],
    ["Texas", "Texas_rucc1", True, "#B8860B", 0, 2015, 2025],
    ["Texas", "Texas_nonrucc1", True, "#F6E8A6", 0, 2015, 2025],
    ["Florida", "Florida_rucc1", True, "#01665E", 0, 2015, 2025],
    ["Florida", "Florida_nonrucc1", True, "#C7EAE5", 0, 2015, 2025],
]

weather_variable='gust'
#
# for group in groups:
#     print(f'Running Bayesian Studies for {group[1]} County, {group[0]}')
#     # step 1: read files
#     merged_data, event_data = read_files.read_files(state=group[0],county=group[1],multi_county=group[2],
#                                                     start_year=group[5],end_year=group[6])
#     # find number of customers in each group
#     if group[2]==False:
#         group[4] = func.get_customers_in_county(group[1], group[0])
#         event_data["customer_hours_norm"]=event_data["customer_hours"]/group[4]
#     else:
#         county_customer_map = {
#             county: func.get_customers_in_county(county, group[0])
#             for county in event_data["counties_affected"].unique()
#         }
#         event_data["customer_hours_norm"] = (
#                 event_data["customer_hours"]
#                 / event_data["counties_affected"].map(county_customer_map)
#         )
#         group[4]=func.get_customers_in_county_group(group[0], merged_data)
#     if 'nonrucc1' in group[1]:
#         type='nonrucc1'
#     else:
#         type='rucc1'
    # func.calculate_saledi(group[0], event_data, type)
    # func.calculate_VAR_CVAR(group[0],group[1], event_data, group[3])
    # # create weather distribution
    # func.setup_weather_distribution(merged_data, group[0], group[1], weather_variable, group[3])
    # make probability maps
    # func.create_probability_mapping(merged_data, group[2], group[0], group[1], weather_variable)
    # # # fit probabilities to bayesian
    # print('Fitting Outage Probability Curve.')
    # func.load_sigmoid(group[0],group[1], group[3], weather_variable)

    # fit regression to bayesian
    # print('Fitting Loss Curve.')
    # func.load_loglinear(group[0],group[1], event_data, group[3], weather_variable)

# func.plot_multiple_wind_profiles(weather_variable, groups)
# func.plot_multiple_outage_probabilities(groups, weather_variable)
# func.plot_multiple_outage_magnitudes(groups, f'max_{weather_variable}')
# func.plot_large_events_profile(groups, weather_variable)
func.event_histograms(groups)

