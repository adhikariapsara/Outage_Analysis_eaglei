import read_files
import func

# groups= [["Texas", "Harris", False, "goldenrod", 0, 2018, 2025],
#         ["Washington", "King", False, "blue", 0, 2015, 2025],
#         ["California", "Los Angeles", False, "red", 0, 2015, 2025],
#         ["Massachusetts", "Suffolk", False, "green", 0, 2015, 2025],
#         ["Illinois", "Cook", False, "black", 0, 2015, 2025],
#         ["Florida", "Miami-Dade", False, "teal", 0, 2015, 2025],
#         ["Arizona", "Maricopa", False, "magenta", 0, 2015, 2025],
#         ["New York", "Kings", False, "brown", 0, 2015, 2025],
# ]
# groups=[
#     ["Illinois", "rural", True, "purple", 0, 2015, 2025],
#     ["Illinois", "Cook", False, "black", 0, 2015, 2025],
#     ["Illinois", "semi_urban", True, "indigo", 0, 2015, 2025],
# ]

groups=[
    ["Washington", "Washington_rural", True, "lightblue", 0, 2015, 2025],
    ["Washington", "King", False, "blue", 0, 2015, 2025],
]
# groups=[
#     ['Washington', 'King', False, 'blue', 0, 2014, 2024],
#     ['Washington','Clallam',False, 'purple', 0, 2020, 2024],
# ]

weather_variable='gust'

for group in groups:
    print(f'Running Bayesian Studies for {group[1]} County, {group[0]}')
    # step 1: read files
    merged_data, event_data = read_files.read_files(state=group[0],county=group[1],multi_county=group[2],
                                                    start_year=group[5],end_year=group[6])
    # find number of customers in each group
    if group[2]==False:
        group[4] = func.get_customers_in_county(group[1], group[0])
        event_data["customer_hours_norm"]=event_data["customer_hours"]/group[4]
    else:
        county_customer_map = {
            county: func.get_customers_in_county(county, group[0])
            for county in event_data["counties_affected"].unique()
        }
        event_data["customer_hours_norm"] = (
                event_data["customer_hours"]
                / event_data["counties_affected"].map(county_customer_map)
        )
        # event_data["customer_hours_norm"]=event_data["customer_hours"]/data_analysis.get_customers_in_county(event_data["counties_affected"], group[0])
        group[4]=func.get_customers_in_county_group(group[0], merged_data)
    # create weather distribution
    func.setup_weather_distribution(merged_data, group[0], group[1], weather_variable, group[3])
    # make probability maps
    func.create_probability_mapping(merged_data, group[2], group[0], group[1], weather_variable)
    # # fit probabilities to bayesian
    print('Fitting Outage Probability Curve.')
    func.load_sigmoid(group[0],group[1], group[3], weather_variable)

    # fit regression to bayesian
    print('Fitting Loss Curve.')
    func.load_loglinear(group[0],group[1], event_data, group[3], weather_variable)

func.plot_multiple_wind_profiles(weather_variable, groups)
func.plot_multiple_outage_probabilities(groups)
func.plot_multiple_outage_magnitudes(groups)
func.plot_large_events_profile(groups, weather_variable)