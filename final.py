import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import io
import openpyxl

#Remove download button from dataframe
st.markdown(
                """
                <style>
                [data-testid="stBaseButton-elementToolbar"] {
                    display: none;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

# Define vehicle types with their characteristics
VEHICLE_TYPES = {
    'Truck': {'capacity': 50, 'max_distance': 100},
    'Van': {'capacity': 30, 'max_distance': 70},
    'Car': {'capacity': 20, 'max_distance': 50},
    'Scooter': {'capacity': 10, 'max_distance': 30}
}

#default vehicle values

default_vehicles_fleet = {
        'Vehicle Type': list(VEHICLE_TYPES.keys()),
        'Quantity': [2] * len(VEHICLE_TYPES),
        'Capacity': [VEHICLE_TYPES[vtype]['capacity'] for vtype in VEHICLE_TYPES],
        'Max Distance': [VEHICLE_TYPES[vtype]['max_distance'] for vtype in VEHICLE_TYPES]
    }


# Initialize session state
if 'demands' not in st.session_state:
    st.session_state.demands = None
if 'pickups' not in st.session_state:
    st.session_state.pickups = None
if 'vehicle_fleet' not in st.session_state:
    st.session_state.vehicle_fleet = pd.DataFrame(default_vehicles_fleet)


def create_sample_excel():
    """Creates a sample Excel file with a single sheet format"""
    # Create sample customer data
    df = pd.DataFrame({
        'customer_id': range(1, 29),  # 28 customers
        'x_coord': [1, 4, 5, 7, 10, 11, 14, 15, 16, 19, 22, 25, 27, 30, 35, 40, -2, -3, -6, -4, 2, 3, 4, 5, 4, 7, 6,
                    11],
        'y_coord': [3, 3, 5, 8, 5, 3, 1, 5, 10, 7, 12, 8, 6, 5, 10, 5, -3, -5, 7, 8, -4, -5, -6, -9, -5, -2, -9, -3],
        'demand': np.random.randint(1, 21, size=28),
        'pickup': np.random.randint(0, 11, size=28)
    })

    # Create Excel file
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='customer_data', index=False)

    return buffer.getvalue()


def load_excel_data(uploaded_file):
    """Loads and validates customer data from the uploaded Excel file"""
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        # Check required columns
        required_columns = ['customer_id', 'x_coord', 'y_coord', 'demand', 'pickup']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return None, [f"Missing columns: {', '.join(missing_cols)}"]

        # Validate data types and values
        if not all(df['demand'] >= 0):
            return None, ["Demands cannot be negative"]
        if not all(df['pickup'] >= 0):
            return None, ["Pickups cannot be negative"]

        # Process the data
        locations = [(0, 0)]  # Start with depot
        locations.extend(list(zip(df['x_coord'], df['y_coord'])))

        demands = df['demand'].tolist()
        pickups = df['pickup'].tolist()

        return {
            'locations': locations,
            'demands': demands,
            'pickups': pickups
        }, None

    except Exception as e:
        return None, [f"Error reading Excel file: {str(e)}"]


def customer_data_editor():
    """Creates an editable interface for customer demands and pickups"""
    st.write("### Customer Demands and Pickups Editor")

    # Define number of customers (excluding depot) - matches the number of locations minus depot
    num_customers = 28  # Total locations minus depot (29-1)

    # Create initial data if not exists
    if st.session_state.demands is None or st.session_state.pickups is None:
        st.session_state.demands = [np.random.randint(1, 21) for _ in range(num_customers)]
        st.session_state.pickups = [np.random.randint(0, 11) for _ in range(num_customers)]

    # Create DataFrame for editing
    df = pd.DataFrame({
        'Customer': range(1, len(st.session_state.demands) + 1),
        'Demand': st.session_state.demands,
        'Pickup': st.session_state.pickups
    })

    # Create editable dataframe
    edited_df = st.data_editor(
        df,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Customer": st.column_config.NumberColumn(
                "Customer",
                help="Customer ID",
                disabled=True,
                format="%d"
            ),
            "Demand": st.column_config.NumberColumn(
                "Demand",
                help="Delivery demand for this customer",
                min_value=0,
                max_value=100,
                step=1,
                format="%d"
            ),
            "Pickup": st.column_config.NumberColumn(
                "Pickup",
                help="Pickup amount for this customer",
                min_value=0,
                max_value=100,
                step=1,
                format="%d"
            )
        }
    )

    # Update session state with edited values
    st.session_state.demands = edited_df['Demand'].tolist()
    st.session_state.pickups = edited_df['Pickup'].tolist()

    # Add buttons for random generation and reset
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Random Values"):
            st.session_state.demands = [np.random.randint(1, 21) for _ in range(num_customers)]
            st.session_state.pickups = [np.random.randint(0, 11) for _ in range(num_customers)]
            st.rerun()

    with col2:
        if st.button("Reset to Default"):
            st.session_state.demands = None
            st.session_state.pickups = None
            st.rerun()


def create_vehicle_fleet_template():
    """Creates a sample Excel file with vehicle fleet template"""
    # Create sample data using the VEHICLE_TYPES dictionary
    df = pd.DataFrame(default_vehicles_fleet)

    # Create Excel file
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='vehicle_fleet', index=False)

    return buffer.getvalue()


def load_vehicle_fleet_data(uploaded_file):
    """Loads and validates vehicle fleet data from the uploaded Excel file"""
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        # Check required columns
        required_columns = ['Vehicle Type', 'Quantity', 'Capacity', 'Max Distance']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return None, [f"Missing columns: {', '.join(missing_cols)}"]

        # Validate data types and values
        if not all(df['Quantity'] >= 0):
            return None, ["Quantities cannot be negative"]
        if not all(df['Capacity'] > 0):
            return None, ["Capacities must be positive"]
        if not all(df['Max Distance'] > 0):
            return None, ["Maximum distances must be positive"]

        # Validate vehicle types
        invalid_types = [vtype for vtype in df['Vehicle Type'] if vtype not in VEHICLE_TYPES]
        if invalid_types:
            return None, [f"Invalid vehicle types: {', '.join(invalid_types)}"]

        return df, None

    except Exception as e:
        return None, [f"Error reading Excel file: {str(e)}"]

def vehicle_fleet_editor():
    """Creates an editable interface for vehicle fleet configuration with Excel upload option"""

    # Create tabs for different input methods
    tabs = st.tabs(["Manual Entry", "File Upload", "Instructions"])

    # Tab 1: Manual Entry
    with tabs[0]:
        st.write("### Vehicle Fleet Configuration")

        # Create editable dataframe
        edited_df = st.data_editor(
            st.session_state.vehicle_fleet,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Vehicle Type": st.column_config.TextColumn(
                    "Vehicle Type",
                    help="Type of vehicle",
                    disabled=True,
                ),
                "Quantity": st.column_config.NumberColumn(
                    "Quantity",
                    help="Number of vehicles of this type",
                    min_value=0,
                    max_value=5,
                    step=1,
                    format="%d"
                ),
                "Capacity": st.column_config.NumberColumn(
                    "Capacity",
                    help="Maximum cargo capacity",
                    format="%d"
                ),
                "Max Distance": st.column_config.NumberColumn(
                    "Max Distance",
                    help="Maximum travel distance",
                    format="%d"
                )
            }
        )

        if not edited_df.equals(st.session_state.vehicle_fleet):
            st.session_state.vehicle_fleet = edited_df

            # Trigger a rerun to update the displayed data
            st.rerun()

        #st.session_state.vehicle_fleet = edited_df

        # Add reset button in manual entry tabs
        if st.button("Reset Fleet"):
            st.session_state.vehicle_fleet = pd.DataFrame(default_vehicles_fleet)
            st.rerun()

    # Tab 2: File Upload
    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Upload Fleet Data")
            uploaded_file = st.file_uploader("Upload your vehicle fleet Excel file",
                                             type=['xlsx'],
                                             key="vehicle_fleet_uploader")

            if uploaded_file is not None:
                with st.spinner("Loading and validating fleet data..."):
                    data, errors = load_vehicle_fleet_data(uploaded_file)
                    if errors:
                        for error in errors:
                            st.error(f"❌ {error}")
                    else:
                        st.success("✅ Fleet data loaded successfully!")
                        # Update session state with uploaded data
                        st.session_state.vehicle_fleet = data
                        # Display the uploaded data
                        st.subheader("📊 Fleet Data Preview")
                        st.dataframe(data)

        with col2:
            st.subheader("📥 Download Template")
            fleet_template = create_vehicle_fleet_template()
            st.download_button(
                label="Download Fleet Template",
                data=fleet_template,
                file_name="vehicle_fleet_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download a sample Excel template for vehicle fleet configuration"
            )

    # Tab 3: Instructions
    with tabs[2]:
        st.subheader("📋 Vehicle Fleet Excel Format Instructions")
        st.markdown("""
            ### Required Columns:
            - **Vehicle Type**: Type of vehicle (must match predefined types)
            - **Quantity**: Number of vehicles of each type
            - **Capacity**: Maximum cargo capacity for each vehicle type
            - **Max Distance**: Maximum travel distance for each vehicle type

            ### Important Notes:
            - Vehicle types must match the predefined types exactly
            - Quantities must be non-negative integers
            - Capacity and Max Distance must be positive numbers
            - Each vehicle type should appear only once

            ### Valid Vehicle Types:
            - Truck
            - Van
            - Car
            - Scooter

            ### Example Data Format:
            | Vehicle Type | Quantity | Capacity | Max Distance |
            |-------------|----------|----------|---------------|
            | Truck       | 2        | 50       | 100           |
            | Van         | 1        | 30       | 70            |
            | Car         | 3        | 20       | 50            |
            | Scooter     | 2        | 10       | 30            |
        """)

def customer_demands_pickup_editor():
    tabs = st.tabs(["Manual Entry", "File Upload", "Instructions"])

    # Tab 1: Manual Entry
    with tabs[0]:
        customer_data_editor()  # Your existing customer_data_editor function

    # Tab 2: File Upload/Download
    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Upload Data")
            uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

            if uploaded_file is not None:
                with st.spinner("Loading and validating data..."):
                    data, errors = load_excel_data(uploaded_file)
                    if errors:
                        for error in errors:
                            st.error(f"❌ {error}")
                    else:
                        st.success("✅ Data loaded successfully!")
                        # Update session state with uploaded data
                        st.session_state.demands = data['demands']
                        st.session_state.pickups = data['pickups']

                        # Display the uploaded data
                        st.subheader("📊 Data Preview")
                        df = pd.read_excel(uploaded_file)
                        st.dataframe(df)

        with col2:
            st.subheader("📥 Download Template")
            sample_excel = create_sample_excel()  # Your existing create_sample_excel function
            st.download_button(
                label="Download Excel Template",
                data=sample_excel,
                file_name="vrp_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download a sample Excel template with the required format"
            )

    # Tab 3: Instructions
    with tabs[2]:
        st.subheader("📋 Excel File Format Instructions")
        st.markdown("""
                        ### Required Columns:
                        - **customer_id**: Unique identifier for each customer
                        - **x_coord**: X coordinate of customer location
                        - **y_coord**: Y coordinate of customer location
                        - **demand**: Delivery demand quantity
                        - **pickup**: Pickup quantity

                        ### Important Notes:
                        - The depot is automatically added at coordinates (0,0)
                        - All demands and pickups must be non-negative numbers
                        - Each customer must have a unique ID
                        - Coordinates can be decimal numbers
                        - Demands and pickups must be integer values

                        ### Example Data Format:
                        | customer_id | x_coord | y_coord | demand | pickup  |
                        |------------ |---------|---------|---------|--------|
                        | 1           | 10.5    | 20.3    | 15      | 5      |
                        | 2           | -5.2    | 8.7     | 22      | 0      |
                        | 3           | 15.0    | -12.4   | 8       | 12     |
                    """)


def create_data_model(distance_type, custom_demands=None, custom_pickups=None):
    """Stores the data for the problem."""
    locations = [
        (0, 0),  # Depot
        (1, 3), (4, 3), (5, 5), (7, 8), (10, 5),
        (11, 3), (14, 1), (15, 5), (16, 10), (19, 7),
        (22, 12), (25, 8), (27, 6), (30, 5), (35, 10),
        (40, 5), (-2, -3), (-3, -5), (-6, 7), (-4, 8),
        (2, -4), (3, -5), (4, -6), (5, -9), (4, -5),
        (7, -2), (6, -9), (11, -3)
    ]

    # Process vehicle fleet data
    vehicle_fleet = []
    vehicle_capacities = []
    vehicle_max_distances = []
    vehicle_types = []  # New list to store vehicle types
    num_vehicles = 0

    for _, row in st.session_state.vehicle_fleet.iterrows():
        if row['Quantity'] > 0:
            vehicle_fleet.append((row['Vehicle Type'], row['Quantity']))
            for _ in range(row['Quantity']):
                vehicle_capacities.append(row['Capacity'])
                vehicle_max_distances.append(row['Max Distance'])
                vehicle_types.append(row['Vehicle Type'])  # Store vehicle type
                num_vehicles += 1

    if custom_demands is None:
        np.random.seed(42)
        demands = [0] + [np.random.randint(1, 21) for _ in range(len(locations) - 1)]
    else:
        demands = [0] + custom_demands

    if custom_pickups is None:
        pickups = [0] + [np.random.randint(0, 11) for _ in range(len(locations) - 1)]
    else:
        pickups = [0] + custom_pickups

    def calculate_distance(p1, p2, type='Euclidean'):
        if type == 'Manhattan':
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        else:  # Euclidean
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    distance_matrix = []
    for i in range(len(locations)):
        row = []
        for j in range(len(locations)):
            if i == j:
                row.append(0)
            else:
                row.append(int(calculate_distance(locations[i], locations[j], type)))
        distance_matrix.append(row)

    return {
        "distance_matrix": distance_matrix,
        "num_vehicles": num_vehicles,
        "depot": 0,
        "locations": locations,
        "demands": demands,
        "pickups": pickups,
        "vehicle_capacities": vehicle_capacities,
        "vehicle_max_distances": vehicle_max_distances,
        "vehicle_fleet": vehicle_fleet,
        "vehicle_types": vehicle_types  # Add vehicle types to the data model
    }


def plot_locations(data):
    st.write("### Customer Delivery and Pickup Locations")
    st.write("""
            The graph below shows the distribution of delivery and pickup locations.
            Each customer location is numbered, and the depot is marked in red.
            The size of each point represents the demand or pickup quantity at that location.
            """)

    plt.figure(figsize=(10, 8))
    xs, ys = zip(*data['locations'])

    # Plot customer locations with sizes based on demands
    demands_normalized = np.array(data['demands'][1:]) * 10  # Scale demands for visualization
    pickups_normalized = np.array(data['pickups'][1:]) * 10  # Scale pickups for visualization
    plt.scatter(xs[1:], ys[1:], c='blue', s=demands_normalized, alpha=0.6,
                label='Customer Deliveries')
    plt.scatter(xs[1:], ys[1:], c='green', s=pickups_normalized, alpha=0.6,
                label='Customer Pickups')

    # Plot depot
    plt.scatter(xs[0], ys[0], c='red', marker='*', s=200,
                edgecolors='black', label='Depot')

    # Add location numbers and demand annotations
    for i, (x, y) in enumerate(data['locations']):
        if i == 0:
            plt.annotate(f'Depot', (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
        else:
            plt.annotate(f'Customer {i}\nDemand: {data["demands"][i]}\nPickup: {data["pickups"][i]}',
                         (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))

    plt.title('Depot, Customer Locations, Demands, and Pickups', pad=20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)


def print_solution(data, manager, routing, solution):
    st.write("### Solution")
    st.write("""
    The optimized routes for each vehicle, including distances, loads, and pickups:""")

    total_distance = 0
    total_load = 0

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        vehicle_type = data["vehicle_types"][vehicle_id]  # Get vehicle type
        plan_output = f"Route for {vehicle_type} {vehicle_id + 1}:\n"  # Use vehicle type in output
        route_distance = 0
        route_load = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index] - data["pickups"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )

        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"

        st.text(plan_output)
        total_distance += route_distance
        total_load += route_load

    st.write(f"Total distance of all routes: {total_distance}m")
    st.write(f"Total load delivered: {total_load}")


def plot_routes(data, manager, routing, solution):
    st.write("### Optimized Vehicle Routes")
    st.write("""
            The graph below shows the optimized routes for each vehicle.
            - The depot is marked with a star
            - Customer locations are shown with circles sized by demand
            - Different colors represent different vehicle routes
            - Arrows show the direction of travel
            """)

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette('husl', n_colors=data['num_vehicles'])

    # Plot all locations
    xs, ys = zip(*data['locations'])
    demands_normalized = np.array(data['demands']) * 10
    pickups_normalized = np.array(data['pickups']) * 10
    plt.scatter(xs[1:], ys[1:], c='lightgray', s=demands_normalized[1:],
                alpha=0.5, zorder=1)
    plt.scatter(xs[0], ys[0], c='red', marker='*', s=200,
                edgecolors='black', zorder=2, label='Depot')

    # Plot routes for each vehicle
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        coordinates = []
        route_load = 0
        vehicle_type = data["vehicle_types"][vehicle_id]  # Get vehicle type

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            coordinates.append(data['locations'][node_index])
            route_load += data['demands'][node_index] - data['pickups'][node_index]
            index = solution.Value(routing.NextVar(index))
        coordinates.append(data['locations'][0])  # Return to depot

        # Plot route with vehicle type in label
        xs, ys = zip(*coordinates)
        plt.plot(xs, ys, '-', color=colors[vehicle_id],
                 label=f'{vehicle_type} {vehicle_id + 1} (Load: {route_load})',
                 linewidth=2, zorder=3)

        # Add direction arrows
        for i in range(len(coordinates) - 1):
            mid_x = (coordinates[i][0] + coordinates[i + 1][0]) / 2
            mid_y = (coordinates[i][1] + coordinates[i + 1][1]) / 2
            plt.annotate('', xy=(coordinates[i + 1][0], coordinates[i + 1][1]),
                         xytext=(mid_x, mid_y),
                         arrowprops=dict(arrowstyle='->', color=colors[vehicle_id]))

    # Add location labels with demands
    for i, (x, y) in enumerate(data['locations']):
        if i == 0:
            plt.annotate(f'Depot', (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
        else:
            plt.annotate(f'Customer {i}\nDemand: {data["demands"][i]}\nPickup: {data["pickups"][i]}',
                         (x, y), xytext=(10, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))

    plt.title('Optimized Vehicle Routes with Demands and Pickups', pad=20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)


def main():
    st.title('Vehicle Routing Problem')
    st.write("""
    This app addresses the Vehicle Routing Problem (VRP) by optimizing 
    the operations of diverse vehicles originating from a single depot. 
    It incorporates key features such as meeting specific customer service requirements, 
    including unique delivery and pickup demands for each customer while adhering to vehicle constraints. 
    The app supports a mixed fleet of vehicle types—trucks, vans, cars, and scooters—with configurable 
    cargo capacities, maximum travel distances, and fleet sizes. 

    Its optimization goals focus on minimizing total travel distance, efficiently assigning routes based on 
    vehicle capabilities, and balancing load distribution across the fleet. 
    Users can customize their fleet composition with the fleet editor, adjust customer demands and pickups 
    through the customer editor, and select their preferred distance calculation method (Euclidean or Manhattan). 
    The solver then generates optimal routes tailored to each vehicle’s constraints and capabilities.
    """)
    st.write("###### Configure your customer demands, pickups, and vehicle fleet to find optimal routes.")
    # Combined data management section
    with st.expander("📊 Customer Demands and Pickups", expanded=True):
        customer_demands_pickup_editor()

    # Vehicle fleet configuration
    with st.expander("📊 Vehicle Fleet Configuration", expanded=True):
        vehicle_fleet_editor()

    # Check if we have any vehicles configured
    total_vehicles = st.session_state.vehicle_fleet['Quantity'].sum()
    if total_vehicles == 0:
        st.warning("Please configure at least one vehicle in the fleet to continue.")
        return

    # Distance type selection
    distance_type = st.selectbox('Distance Metric', ['Euclidean', 'Manhattan'])

    # Create data model with custom demands and pickups
    data = create_data_model(
        distance_type,
        st.session_state.demands,
        st.session_state.pickups
    )

    # Plot initial locations
    plot_locations(data)

    # Add a button to start the algorithm
    if st.button("Optimize Routes"):
        with st.spinner("Calculating optimal routes..."):
            try:
                # Create the routing index manager
                manager = pywrapcp.RoutingIndexManager(
                    len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
                )

                # Create Routing Model
                routing = pywrapcp.RoutingModel(manager)

                # Distance callback
                def distance_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    return data["distance_matrix"][from_node][to_node]

                transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

                # Demand callback
                def demand_callback(from_index):
                    from_node = manager.IndexToNode(from_index)
                    return data["demands"][from_node] - data["pickups"][from_node]

                demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

                # Add capacity constraint for each vehicle
                routing.AddDimensionWithVehicleCapacity(
                    demand_callback_index,
                    0,  # null capacity slack
                    data["vehicle_capacities"],  # vehicle maximum capacities
                    True,  # start cumul to zero
                    'Capacity')

                # Distance constraint
                dimension_name = 'Distance'
                routing.AddDimension(
                    transit_callback_index,
                    0,  # no slack
                    max(data["vehicle_max_distances"]),  # maximum travel distance
                    True,  # start cumul to zero
                    dimension_name)

                distance_dimension = routing.GetDimensionOrDie(dimension_name)

                # Set vehicle-specific distance limits
                for vehicle_id in range(data["num_vehicles"]):
                    index = routing.End(vehicle_id)
                    distance_dimension.SetCumulVarSoftUpperBound(
                        index, data["vehicle_max_distances"][vehicle_id], 100)

                # Set search parameters
                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
                search_parameters.local_search_metaheuristic = (
                    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
                search_parameters.time_limit.FromSeconds(10)

                # Solve and display solution
                solution = routing.SolveWithParameters(search_parameters)

                if solution:
                    print_solution(data, manager, routing, solution)
                    plot_routes(data, manager, routing, solution)
                else:
                    st.error("""No solution found! Try:
                            1. Adding more vehicles
                            2. Adding vehicles with larger capacity
                            3. Reducing customer demands
                            4. Adjusting pickup quantities""")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
