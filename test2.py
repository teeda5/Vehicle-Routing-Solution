import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import io
import openpyxl
from geopy.distance import geodesic

# Remove download button from dataframe
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
    'Truck': {'capacity': 100, 'max_distance': 100},
    'Van': {'capacity': 50, 'max_distance': 70},
    'Car': {'capacity': 20, 'max_distance': 50},
    'Scooter': {'capacity': 10, 'max_distance': 30}
}

# default vehicle values
default_vehicles_fleet = {
    'Vehicle Type': list(VEHICLE_TYPES.keys()),
    'Quantity': [2] * len(VEHICLE_TYPES),
    'Capacity': [VEHICLE_TYPES[vtype]['capacity'] for vtype in VEHICLE_TYPES],
    'Max Distance': [VEHICLE_TYPES[vtype]['max_distance'] for vtype in VEHICLE_TYPES]
}

# Initialize session state
if 'depot_location' not in st.session_state:
    st.session_state.depot_location = (40.7128, -74.0060)  # Default NYC location
if 'available_depots' not in st.session_state:
    st.session_state.available_depots = None
if 'demands' not in st.session_state:
    st.session_state.demands = None
if 'pickups' not in st.session_state:
    st.session_state.pickups = None
if 'vehicle_fleet' not in st.session_state:
    st.session_state.vehicle_fleet = pd.DataFrame(default_vehicles_fleet)


def depot_location_manager():
    """Manages depot location input and selection with improved UI consistency"""
    st.write("### Depot Location Configuration")

    # Create tabs for different input methods
    tabs = st.tabs(["Manual Entry", "File Upload", "Instructions"])

    # Tab 1: Manual Entry
    with tabs[0]:
        st.subheader("📍 Manual Depot Configuration")
        st.write("Enter the coordinates for your depot location:")

        col1, col2 = st.columns(2)
        with col1:
            depot_lat = st.number_input(
                "Depot Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=st.session_state.depot_location[0],
                format="%.6f",
                help="Enter the latitude coordinate for the depot (-90° to 90°)"
            )

            # Add validation feedback for latitude
            if depot_lat < -90 or depot_lat > 90:
                st.error("⚠️ Latitude must be between -90° and 90°")

        with col2:
            depot_lon = st.number_input(
                "Depot Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.depot_location[1],
                format="%.6f",
                help="Enter the longitude coordinate for the depot (-180° to 180°)"
            )

            # Add validation feedback for longitude
            if depot_lon < -180 or depot_lon > 180:
                st.error("⚠️ Longitude must be between -180° and 180°")

        # Update session state
        if -90 <= depot_lat <= 90 and -180 <= depot_lon <= 180:
            st.session_state.depot_location = (depot_lat, depot_lon)

            # Show current depot location on an enhanced map
            st.write("#### Current Depot Location")
            fig = plt.figure(figsize=(10, 8))
            plt.scatter(depot_lon, depot_lat, c='red', marker='*', s=300, label='Depot')

            # Add circle to show approximate service area
            circle = plt.Circle((depot_lon, depot_lat), 0.1, color='red', fill=False, alpha=0.3)
            plt.gca().add_artist(circle)

            plt.title('Depot Location with Service Area')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            # Add coordinates annotation
            plt.annotate(
                f'Depot\nLat: {depot_lat:.6f}\nLon: {depot_lon:.6f}',
                (depot_lon, depot_lat),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='red', alpha=0.7)
            )

            st.pyplot(fig)

    # Tab 2: File Upload
    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Upload Depot Data")
            st.write("Upload an Excel file containing depot locations.")

            uploaded_file = st.file_uploader(
                "Upload depot locations Excel file",
                type=['xlsx'],
                key='depot_uploader',
                help="Upload an Excel file with depot locations data"
            )

            if uploaded_file is not None:
                try:
                    depot_df = pd.read_excel(uploaded_file)

                    # Validate depot data
                    required_columns = ['depot_id', 'latitude', 'longitude', 'name']
                    if not all(col in depot_df.columns for col in required_columns):
                        st.error("❌ Excel file must contain: depot_id, latitude, longitude, and name columns")
                        return

                    # Validate coordinate ranges
                    if not all((-90 <= depot_df['latitude']) & (depot_df['latitude'] <= 90)):
                        st.error("❌ Latitude values must be between -90° and 90°")
                        return
                    if not all((-180 <= depot_df['longitude']) & (depot_df['longitude'] <= 180)):
                        st.error("❌ Longitude values must be between -180° and 180°")
                        return

                    # Store available depots
                    st.session_state.available_depots = depot_df.to_dict('records')

                    # Create radio buttons for depot selection with improved formatting
                    st.write("#### Select Depot Location")
                    depot_options = [f"Depot {row['depot_id']}: {row['name']}" for _, row in depot_df.iterrows()]
                    selected_option = st.radio(
                        "Choose a depot:",
                        options=depot_options,
                        help="Select which depot to use for route optimization",
                        key="depot_radio"
                    )

                    # Extract depot_id from selected option
                    selected_depot_id = int(selected_option.split(':')[0].replace('Depot ', ''))

                    # Update depot location based on selection
                    selected_row = depot_df[depot_df['depot_id'] == selected_depot_id].iloc[0]
                    st.session_state.depot_location = (selected_row['latitude'], selected_row['longitude'])

                    # Show enhanced map with all depots
                    st.write("#### Depot Locations Map")
                    fig = plt.figure(figsize=(10, 8))

                    # Plot all depots
                    plt.scatter(depot_df['longitude'], depot_df['latitude'],
                                c='blue', marker='o', s=100, alpha=0.5, label='Available Depots')

                    # Highlight selected depot
                    plt.scatter(selected_row['longitude'], selected_row['latitude'],
                                c='red', marker='*', s=300, label='Selected Depot')

                    # Add labels for all depots
                    for _, depot in depot_df.iterrows():
                        plt.annotate(
                            f"Depot {depot['depot_id']}: {depot['name']}",
                            (depot['longitude'], depot['latitude']),
                            xytext=(5, 5),
                            textcoords='offset points',
                            bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7)
                        )

                    plt.title('Available Depots and Selection')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)

                    # Display depot information
                    st.write("#### Selected Depot Details")
                    st.json({
                        "Depot ID": int(selected_row['depot_id']),
                        "Name": selected_row['name'],
                        "Latitude": float(selected_row['latitude']),
                        "Longitude": float(selected_row['longitude'])
                    })

                except Exception as e:
                    st.error(f"❌ Error processing depot file: {str(e)}")

        with col2:
            st.subheader("📥 Download Template")
            st.write("Download a template file for depot locations.")

            if st.button("Download Depot Template"):
                depot_template = create_depot_template()
                st.download_button(
                    label="Download Depot Template Excel",
                    data=depot_template,
                    file_name="depot_locations_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # Add template preview
            st.write("#### Template Preview")
            st.dataframe(pd.DataFrame({
                'depot_id': [1, 2, 3],
                'name': ['Main Warehouse', 'Downtown Hub', 'Airport Center'],
                'latitude': [40.7128, 40.7589, 40.6413],
                'longitude': [-74.0060, -73.9851, -73.7781]
            }))

    # Tab 3: Instructions
    with tabs[2]:
        st.subheader("📋 Depot Configuration Guide")

        # General Instructions
        st.write("### How to Configure Depot Location")
        st.markdown("""
        1. **Manual Entry**:
           - Enter the latitude and longitude coordinates directly
           - Values are automatically validated
           - View the location on the interactive map

        2. **File Upload**:
           - Download the template Excel file
           - Fill in your depot locations data
           - Upload the completed file
           - Select the desired depot using the radio buttons
        """)

        # Data Format
        st.write("### Excel File Format")
        st.markdown("""
        Your Excel file should contain the following columns:
        - **depot_id**: Unique identifier for each depot
        - **name**: Descriptive name for the depot
        - **latitude**: Location coordinate (-90° to 90°)
        - **longitude**: Location coordinate (-180° to 180°)
        """)

        # Add coordinate system explanation
        st.write("### Coordinate System Guide")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Latitude Guidelines:**")
            st.markdown("""
            - Range: -90° to 90°
            - Positive: North of equator
            - Negative: South of equator
            - Example: 40.7128° (New York)
            """)

        with col2:
            st.write("**Longitude Guidelines:**")
            st.markdown("""
            - Range: -180° to 180°
            - Positive: East of prime meridian
            - Negative: West of prime meridian
            - Example: -74.0060° (New York)
            """)

        # Add troubleshooting tips
        st.warning("""
        ⚠️ **Troubleshooting Tips**:
        1. Ensure coordinates are in decimal degrees format
        2. Verify coordinates on a map if unsure
        3. Check for any special characters in the Excel file
        4. Make sure all required columns are present
        """)


def create_depot_template():
    """Creates a template Excel file for depot locations"""
    sample_data = {
        'depot_id': [1, 2, 3],
        'name': ['Main Warehouse', 'Downtown Hub', 'Airport Center'],
        'latitude': [40.7128, 40.7589, 40.6413],
        'longitude': [-74.0060, -73.9851, -73.7781]
    }
    df = pd.DataFrame(sample_data)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='depot_locations', index=False)

    return buffer.getvalue()


def create_sample_excel():
    """Creates a sample Excel file with a single sheet format using lat/long coordinates"""
    # Create sample customer data with realistic lat/long coordinates (around a central point)
    base_lat, base_lon = 40.7128, -74.0060  # New York City as example center
    np.random.seed(42)
    num_customers = 28

    # Generate random offsets (in degrees) from the base coordinates
    lat_offsets = np.random.uniform(-0.1, 0.1, num_customers)
    lon_offsets = np.random.uniform(-0.1, 0.1, num_customers)

    df = pd.DataFrame({
        'customer_id': range(1, num_customers + 1),
        'latitude': base_lat + lat_offsets,
        'longitude': base_lon + lon_offsets,
        'demand': np.random.randint(1, 21, size=num_customers),
        'pickup': np.random.randint(0, 11, size=num_customers)
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='customer_data', index=False)

    return buffer.getvalue()


def load_excel_data(uploaded_file):
    """Loads and validates customer data from the uploaded Excel file"""
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        # Check required columns
        required_columns = ['customer_id', 'latitude', 'longitude', 'demand', 'pickup']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return None, [f"Missing columns: {', '.join(missing_cols)}"]

        # Validate data types and values
        if not all(df['demand'] >= 0):
            return None, ["Demands cannot be negative"]
        if not all(df['pickup'] >= 0):
            return None, ["Pickups cannot be negative"]

        # Validate latitude/longitude ranges
        if not all((-90 <= df['latitude']) & (df['latitude'] <= 90)):
            return None, ["Latitude must be between -90 and 90 degrees"]
        if not all((-180 <= df['longitude']) & (df['longitude'] <= 180)):
            return None, ["Longitude must be between -180 and 180 degrees"]

        # Process the data
        locations = [(40.7128, -74.0060)]  # Depot at NYC
        locations.extend(list(zip(df['latitude'], df['longitude'])))

        demands = df['demand'].tolist()
        pickups = df['pickup'].tolist()

        # Update session state with the new data
        st.session_state.demands = demands
        st.session_state.pickups = pickups
        st.session_state.latitudes = df['latitude'].tolist()
        st.session_state.longitudes = df['longitude'].tolist()

        return {
            'locations': locations,
            'demands': demands,
            'pickups': pickups
        }, None

    except Exception as e:
        return None, [f"Error reading Excel file: {str(e)}"]


def customer_data_editor():
    """Creates an editable interface for customer demands and pickups with coordinates"""
    st.write("### Customer Demands and Pickups Editor")

    # Define number of customers (excluding depot)
    num_customers = 28

    # Initialize base coordinates (NYC)
    base_lat, base_lon = 40.7128, -74.0060

    # Create initial data if not exists
    if ('demands' not in st.session_state or
            'pickups' not in st.session_state or
            'latitudes' not in st.session_state or
            'longitudes' not in st.session_state):
        np.random.seed(42)
        lat_offsets = np.random.uniform(-0.1, 0.1, num_customers)
        lon_offsets = np.random.uniform(-0.1, 0.1, num_customers)

        st.session_state.demands = [np.random.randint(1, 21) for _ in range(num_customers)]
        st.session_state.pickups = [np.random.randint(0, 11) for _ in range(num_customers)]
        st.session_state.latitudes = [base_lat + offset for offset in lat_offsets]
        st.session_state.longitudes = [base_lon + offset for offset in lon_offsets]

    # Create DataFrame for editing
    df = pd.DataFrame({
        'Customer': range(1, num_customers + 1),
        'Latitude': st.session_state.latitudes,
        'Longitude': st.session_state.longitudes,
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
            "Latitude": st.column_config.NumberColumn(
                "Latitude",
                help="Customer location latitude",
                min_value=-90,
                max_value=90,
                format="%.6f"
            ),
            "Longitude": st.column_config.NumberColumn(
                "Longitude",
                help="Customer location longitude",
                min_value=-180,
                max_value=180,
                format="%.6f"
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
    st.session_state.latitudes = edited_df['Latitude'].tolist()
    st.session_state.longitudes = edited_df['Longitude'].tolist()

    # Add buttons for random generation and reset
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Random Values"):
            np.random.seed(None)  # Reset seed for true randomness
            lat_offsets = np.random.uniform(-0.1, 0.1, num_customers)
            lon_offsets = np.random.uniform(-0.1, 0.1, num_customers)

            st.session_state.demands = [np.random.randint(1, 21) for _ in range(num_customers)]
            st.session_state.pickups = [np.random.randint(0, 11) for _ in range(num_customers)]
            st.session_state.latitudes = [base_lat + offset for offset in lat_offsets]
            st.session_state.longitudes = [base_lon + offset for offset in lon_offsets]
            st.rerun()

    with col2:
        if st.button("Reset to Default"):
            st.session_state.demands = None
            st.session_state.pickups = None
            st.session_state.latitudes = None
            st.session_state.longitudes = None
            st.rerun()


def customer_demands_pickup_editor():
    """Creates an interface for editing customer demands and pickups with manual and file upload options"""
    tabs = st.tabs(["Manual Entry", "File Upload", "Instructions"])

    # Tab 1: Manual Entry
    with tabs[0]:
        customer_data_editor()  # Your existing customer_data_editor function

    # Tab 2: File Upload/Download
    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Upload Data")
            uploaded_file = st.file_uploader(
                "Upload your Excel file",
                type=['xlsx'],
                key='customer_data_uploader',
                help="Upload an Excel file with customer data"
            )

            # Handle file upload and session state
            if uploaded_file is not None:
                with st.spinner("Loading and validating data..."):
                    data, errors = load_excel_data(uploaded_file)
                    if errors:
                        for error in errors:
                            st.error(f"❌ {error}")
                    else:
                        st.success("✅ Data loaded successfully!")
                        # Update session state with uploaded data
                        if data:
                            st.session_state.demands = data['demands']
                            st.session_state.pickups = data['pickups']
                            locations = data['locations']
                            # Extract latitudes and longitudes from locations
                            st.session_state.latitudes = [loc[0] for loc in locations[1:]]  # Skip depot
                            st.session_state.longitudes = [loc[1] for loc in locations[1:]]  # Skip depot

                        # Display the uploaded data
                        st.subheader("📊 Data Preview")
                        df = pd.read_excel(uploaded_file)
                        st.dataframe(df)
            else:
                # If no file is uploaded (including after discard), ensure session state is properly initialized
                if ('demands' not in st.session_state or
                        st.session_state.demands is None or
                        'pickups' not in st.session_state or
                        st.session_state.pickups is None):
                    # Initialize with default values
                    num_customers = 28
                    np.random.seed(42)

                    # Generate random demands and pickups
                    st.session_state.demands = [np.random.randint(1, 21) for _ in range(num_customers)]
                    st.session_state.pickups = [np.random.randint(0, 11) for _ in range(num_customers)]

                    # Generate random locations around NYC
                    base_lat, base_lon = 40.7128, -74.0060
                    lat_offsets = np.random.uniform(-0.1, 0.1, num_customers)
                    lon_offsets = np.random.uniform(-0.1, 0.1, num_customers)
                    st.session_state.latitudes = [base_lat + offset for offset in lat_offsets]
                    st.session_state.longitudes = [base_lon + offset for offset in lon_offsets]

        with col2:
            st.subheader("📥 Download Template")
            sample_excel = create_sample_excel()
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
            - **latitude**: Latitude coordinate of customer location (-90 to 90 degrees)
            - **longitude**: Longitude coordinate of customer location (-180 to 180 degrees)
            - **demand**: Delivery demand quantity
            - **pickup**: Pickup quantity

            ### Important Notes:
            - The depot is automatically set at coordinates (40.7128, -74.0060) [New York City]
            - Latitude must be between -90 and 90 degrees
            - Longitude must be between -180 and 180 degrees
            - All demands and pickups must be non-negative numbers
            - Each customer must have a unique ID
            - Demands and pickups must be integer values
            - Coordinates should be decimal numbers with up to 6 decimal places

            ### Example Data Format:
            | customer_id | latitude  | longitude  | demand | pickup |
            |------------|-----------|------------|---------|---------|
            | 1          | 40.7589   | -73.9851   | 15      | 5      |
            | 2          | 40.7127   | -74.0059   | 22      | 0      |
            | 3          | 40.7214   | -73.9959   | 8       | 12     |

            ### Coordinate Guidelines:
            - Use decimal degrees format (e.g., 40.7128 not 40°42'46"N)
            - Positive latitude values indicate North, negative indicate South
            - Positive longitude values indicate East, negative indicate West
            - Ensure coordinates are within reasonable distance from depot
            - Verify coordinates on a map if unsure of location accuracy
        """)

        # Add a note about data validation
        st.info("""
            ℹ️ **Note**: The application performs automatic validation of your uploaded data:
            - Checks for required columns
            - Validates coordinate ranges
            - Ensures non-negative demands and pickups
            - Verifies data types and formats
        """)

        # Add troubleshooting tips
        st.warning("""
            ⚠️ **Troubleshooting Tips**:
            1. If your file fails to upload, check the file format and column names
            2. Ensure all values are within the specified ranges
            3. Remove any special characters or formatting from the Excel file
            4. Make sure there are no empty cells in required columns
        """)


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
    """Creates an editable interface for vehicle fleet configuration"""
    st.title("Vehicle Fleet Configuration")

    # Initialize session state if not exists
    if 'vehicle_fleet' not in st.session_state:
        st.session_state.vehicle_fleet = pd.DataFrame(default_vehicles_fleet)

    # Create tabs for different input methods
    tabs = st.tabs(["Manual Entry", "File Upload", "Instructions"])

    # Tab 1: Manual Entry
    with tabs[0]:
        st.write("### Configure Your Vehicle Fleet")
        st.write("Adjust the quantities and characteristics of each vehicle type:")

        # Create editable dataframe with improved styling
        edited_df = st.data_editor(
            st.session_state.vehicle_fleet,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Vehicle Type": st.column_config.TextColumn(
                    "Vehicle Type",
                    help="Type of vehicle",
                    disabled=True,
                    width="medium"
                ),
                "Quantity": st.column_config.NumberColumn(
                    "Quantity",
                    help="Number of vehicles of this type",
                    min_value=0,
                    max_value=10,
                    step=1,
                    format="%d",
                    width="small"
                ),
                "Capacity": st.column_config.NumberColumn(
                    "Capacity",
                    help="Maximum cargo capacity",
                    min_value=1,
                    format="%d",
                    width="medium"
                ),
                "Max Distance": st.column_config.NumberColumn(
                    "Max Distance",
                    help="Maximum travel distance (km)",
                    min_value=1,
                    format="%d",
                    width="medium"
                )
            }
        )

        # Update session state if changes are made
        if not edited_df.equals(st.session_state.vehicle_fleet):
            st.session_state.vehicle_fleet = edited_df

        # Display fleet statistics
        total_vehicles = edited_df['Quantity'].sum()
        total_capacity = (edited_df['Quantity'] * edited_df['Capacity']).sum()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Vehicles", total_vehicles)
        with col2:
            st.metric("Total Fleet Capacity", total_capacity)

        # Validation feedback
        if total_vehicles == 0:
            st.warning("⚠️ Please add at least one vehicle to the fleet")
        elif total_vehicles > 20:
            st.warning("⚠️ Large number of vehicles may increase computation time")

        # Reset button
        if st.button("Reset to Default Configuration"):
            st.session_state.vehicle_fleet = pd.DataFrame(default_vehicles_fleet)
            st.rerun()

    # Tab 2: File Upload
    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Upload Fleet Data")
            uploaded_file = st.file_uploader(
                "Upload your vehicle fleet Excel file",
                type=['xlsx'],
                key="vehicle_fleet_uploader",
                help="Upload an Excel file with your vehicle fleet configuration"
            )

            if uploaded_file is not None:
                with st.spinner("Loading and validating fleet data..."):
                    data, errors = load_vehicle_fleet_data(uploaded_file)
                    if errors:
                        for error in errors:
                            st.error(f"❌ {error}")
                    else:
                        st.success("✅ Fleet data loaded successfully!")
                        st.session_state.vehicle_fleet = data
                        st.dataframe(data, use_container_width=True)

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
        st.subheader("📋 Vehicle Fleet Configuration Guide")

        # General Instructions
        st.write("### How to Configure Your Vehicle Fleet")
        st.markdown("""
        1. **Manual Entry**:
           - Adjust vehicle quantities using the number input
           - Modify capacity and max distance values as needed
           - Changes are saved automatically

        2. **File Upload**:
           - Download the template Excel file
           - Fill in your fleet configuration
           - Upload the completed file
        """)

        # Data Format
        st.write("### Excel File Format")
        st.markdown("""
        Your Excel file should contain the following columns:
        - **Vehicle Type**: Must be one of: Truck, Van, Car, Scooter
        - **Quantity**: Integer number of vehicles (0-10)
        - **Capacity**: Positive integer representing cargo capacity
        - **Max Distance**: Maximum travel distance in kilometers
        """)

        # Example Table
        st.write("### Example Configuration")
        example_data = pd.DataFrame({
            'Vehicle Type': ['Truck', 'Van', 'Car', 'Scooter'],
            'Quantity': [2, 3, 1, 1],
            'Capacity': [100, 50, 20, 10],
            'Max Distance': [100, 70, 50, 30]
        })
        st.dataframe(example_data, use_container_width=True)

        # Add a note about data validation
        st.info("""
                    ℹ️ **Note**: The application performs automatic validation of your uploaded data:
                    - Checks for required columns
                    - Validates vehicle type
                    - Ensures non-negative capacity and maximum distance
                    - Verifies data types and formats
                """)

        # Add troubleshooting tips
        st.warning("""
                    ⚠️ **Troubleshooting Tips**:
                    1. If your file fails to upload, check the file format and column names
                    2. Ensure all values are within the specified ranges
                    3. Remove any special characters or formatting from the Excel file
                    4. Make sure there are no empty cells in required columns

                    🔧 If problems persist, try resetting to default configuration and gradually make changes.
                """)


def create_data_model(distance_type, custom_demands=None, custom_pickups=None):
    """Stores the data for the problem using lat/long coordinates."""
    # Use selected depot location instead of hardcoded NYC coordinates
    depot_lat, depot_lon = st.session_state.depot_location
    np.random.seed(42)
    num_locations = 28

    # Generate random offsets (in degrees) from the base coordinates
    lat_offsets = np.random.uniform(-0.1, 0.1, num_locations)
    lon_offsets = np.random.uniform(-0.1, 0.1, num_locations)

    locations = [(depot_lat, depot_lon)]  # Depot
    locations.extend([(depot_lat + lat_offset, depot_lon + lon_offset)
                      for lat_offset, lon_offset in zip(lat_offsets, lon_offsets)])

    # Process vehicle fleet data
    vehicle_fleet = []
    vehicle_capacities = []
    vehicle_max_distances = []
    vehicle_types = []
    num_vehicles = 0

    for _, row in st.session_state.vehicle_fleet.iterrows():
        if row['Quantity'] > 0:
            vehicle_fleet.append((row['Vehicle Type'], row['Quantity']))
            for _ in range(row['Quantity']):
                # Convert capacity to handle both deliveries and pickups
                vehicle_capacities.append(int(row['Capacity'] * 1.5))  # Increased capacity buffer
                vehicle_max_distances.append(int(row['Max Distance'] * 1000))  # Convert to meters
                vehicle_types.append(row['Vehicle Type'])
                num_vehicles += 1

    if num_vehicles == 0:
        raise ValueError("No vehicles configured. Please add at least one vehicle.")

    # Validate and process demands and pickups
    if custom_demands is None:
        demands = [0] + [np.random.randint(1, 21) for _ in range(len(locations) - 1)]
    else:
        demands = [0] + custom_demands

    if custom_pickups is None:
        pickups = [0] + [np.random.randint(0, 11) for _ in range(len(locations) - 1)]
    else:
        pickups = [0] + custom_pickups

    # Validate total load doesn't exceed total vehicle capacity
    total_demand = sum(demands)
    total_pickup = sum(pickups)
    total_capacity = sum(vehicle_capacities)

    if total_demand + total_pickup > total_capacity:
        raise ValueError(
            f"Total load ({total_demand + total_pickup}) exceeds total vehicle capacity ({total_capacity})")

    def calculate_distance(coord1, coord2):
        """Calculate distance between two lat/long coordinates in kilometers"""
        return int(geodesic(coord1, coord2).kilometers * 1000)  # Convert to meters

    # Create distance matrix using actual geographic distances
    distance_matrix = []
    for i in range(len(locations)):
        row = []
        for j in range(len(locations)):
            if i == j:
                row.append(0)
            else:
                row.append(calculate_distance(locations[i], locations[j]))
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
        "vehicle_types": vehicle_types
    }


def plot_locations(data):
    """Plot customer locations using lat/long coordinates"""
    st.write("### Customer Delivery and Pickup Locations")
    st.write("""
            The graph below shows the distribution of delivery and pickup locations.
            Each customer location is numbered, and the depot is marked in red.
            The size of each point represents the demand or pickup quantity at that location.
            """)

    plt.figure(figsize=(12, 8))
    lats, lons = zip(*data['locations'])

    # Plot customer locations with sizes based on demands
    demands_normalized = np.array(data['demands'][1:]) * 10
    pickups_normalized = np.array(data['pickups'][1:]) * 10
    plt.scatter(lons[1:], lats[1:], c='blue', s=demands_normalized, alpha=0.6,
                label='Customer Deliveries')
    plt.scatter(lons[1:], lats[1:], c='green', s=pickups_normalized, alpha=0.6,
                label='Customer Pickups')

    # Plot depot
    plt.scatter(lons[0], lats[0], c='red', marker='*', s=200,
                edgecolors='black', label='Depot')

    # Add location numbers and demand annotations
    for i, (lat, lon) in enumerate(data['locations']):
        if i == 0:
            plt.annotate(f'Depot', (lon, lat), xytext=(10, 10),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
        else:
            plt.annotate(f'Customer {i}\nDemand: {data["demands"][i]}\nPickup: {data["pickups"][i]}',
                         (lon, lat), xytext=(10, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))

    plt.title('Depot and Customer Locations (Lat/Long)', pad=20)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
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
    """Plot optimized routes using lat/long coordinates"""
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
    lats, lons = zip(*data['locations'])
    demands_normalized = np.array(data['demands']) * 10
    pickups_normalized = np.array(data['pickups']) * 10
    plt.scatter(lons[1:], lats[1:], c='lightgray', s=demands_normalized[1:],
                alpha=0.5, zorder=1)
    plt.scatter(lons[0], lats[0], c='red', marker='*', s=200,
                edgecolors='black', zorder=2, label='Depot')

    # Plot routes for each vehicle
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        coordinates = []
        route_load = 0
        vehicle_type = data["vehicle_types"][vehicle_id]

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            coordinates.append(data['locations'][node_index])
            route_load += data['demands'][node_index] - data['pickups'][node_index]
            index = solution.Value(routing.NextVar(index))
        coordinates.append(data['locations'][0])

        # Plot route
        lats, lons = zip(*coordinates)
        plt.plot(lons, lats, '-', color=colors[vehicle_id],
                 label=f'{vehicle_type} {vehicle_id + 1} (Load: {route_load})',
                 linewidth=2, zorder=3)

        # Add direction arrows
        for i in range(len(coordinates) - 1):
            mid_lon = (coordinates[i][1] + coordinates[i + 1][1]) / 2
            mid_lat = (coordinates[i][0] + coordinates[i + 1][0]) / 2
            plt.annotate('', xy=(coordinates[i + 1][1], coordinates[i + 1][0]),
                         xytext=(mid_lon, mid_lat),
                         arrowprops=dict(arrowstyle='->', color=colors[vehicle_id]))

    # Add location labels
    for i, (lat, lon) in enumerate(data['locations']):
        if i == 0:
            plt.annotate(f'Depot', (lon, lat), xytext=(10, 10),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
        else:
            plt.annotate(f'Customer {i}\nDemand: {data["demands"][i]}\nPickup: {data["pickups"][i]}',
                         (lon, lat), xytext=(10, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))

    plt.title('Optimized Vehicle Routes (Lat/Long)', pad=20)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)


def calculate_total_distance(routing, solution, manager, data):
    """Calculate total distance of all routes."""
    total_distance = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        total_distance += route_distance
    return total_distance


def main():
    st.title('Vehicle Routing Problem Solver')

    # Application description
    st.write("""
    This application solves the Vehicle Routing Problem (VRP) with the following features:
    - Multiple vehicle types with different capacities
    - Delivery and pickup demands for each customer
    - Distance constraints for vehicles
    - Geographic distance calculations
    - Route optimization for minimal total distance
    """)

    # Create main tabs
    tabs = st.tabs(["Depot Configuration", "Customer Data Input", "Vehicle Configuration", "Route Optimization"])

    # Data Input Tab
    with tabs[0]:
        st.header("Depot Data Input")
        with st.expander("📍 Depot Location", expanded=True):
            depot_location_manager()

    # Data Input Tab
    with tabs[1]:
        st.header("Customer Data Input")
        with st.expander("📊 Customer Demands and Pickups", expanded=True):
            customer_demands_pickup_editor()

    # Configuration Tab
    with tabs[2]:
        st.header("Vehicle Fleet Configuration")
        with st.expander("🚛 Vehicle Fleet Setup", expanded=True):
            vehicle_fleet_editor()

        # Validate vehicle configuration
        total_vehicles = st.session_state.vehicle_fleet['Quantity'].sum()
        if total_vehicles == 0:
            st.warning("⚠️ Please configure at least one vehicle in the fleet to continue.")
            return

        # Distance metric selection
        st.subheader("Distance Calculation Method")
        distance_type = st.selectbox(
            'Select distance metric:',
            ['Euclidean', 'Manhattan'],
            help="Choose how distances between locations should be calculated"
        )

    # Optimization Tab
    with tabs[3]:
        st.header("Route Optimization")

        if st.button("🎯 Optimize Routes", help="Click to calculate optimal routes"):
            with st.spinner("Calculating optimal routes..."):
                try:
                    # Create and validate data model
                    data = create_data_model(
                        distance_type,
                        st.session_state.demands,
                        st.session_state.pickups
                    )

                    # Validate capacity constraints
                    total_demand = sum(data['demands'])
                    total_pickup = sum(data['pickups'])
                    total_capacity = sum(data['vehicle_capacities'])

                    if total_demand + total_pickup > total_capacity:
                        st.error(f"""
                        ❌ Total load ({total_demand + total_pickup}) exceeds total vehicle capacity ({total_capacity}).
                        Please either:
                        - Increase vehicle capacity
                        - Add more vehicles
                        - Reduce customer demands/pickups
                        """)
                        return

                    # Create routing model components
                    manager = pywrapcp.RoutingIndexManager(
                        len(data["distance_matrix"]),
                        data["num_vehicles"],
                        data["depot"]
                    )
                    routing = pywrapcp.RoutingModel(manager)

                    # Distance callback
                    def distance_callback(from_index, to_index):
                        from_node = manager.IndexToNode(from_index)
                        to_node = manager.IndexToNode(to_index)
                        return data["distance_matrix"][from_node][to_node]

                    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

                    # Add Distance dimension
                    routing.AddDimension(
                        transit_callback_index,
                        0,  # no slack
                        max(data["vehicle_max_distances"]),
                        True,  # start cumul to zero
                        'Distance'
                    )

                    # Demand callback
                    def demand_callback(from_index):
                        from_node = manager.IndexToNode(from_index)
                        return data["demands"][from_node] + data["pickups"][from_node]

                    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

                    # Add Capacity dimension
                    routing.AddDimensionWithVehicleCapacity(
                        demand_callback_index,
                        0,  # null capacity slack
                        data["vehicle_capacities"],
                        True,  # start cumul to zero
                        'Capacity'
                    )

                    # Set search parameters
                    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                    search_parameters.first_solution_strategy = (
                        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
                    )
                    search_parameters.local_search_metaheuristic = (
                        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
                    )
                    search_parameters.time_limit.FromSeconds(30)
                    search_parameters.solution_limit = 100

                    # Solve the problem
                    solution = routing.SolveWithParameters(search_parameters)

                    if solution:
                        st.success("✅ Optimal routes found!")

                        # Create solution view tabs
                        solution_tabs = st.tabs(["Route Details", "Map View", "Statistics"])

                        with solution_tabs[0]:
                            print_solution(data, manager, routing, solution)

                        with solution_tabs[1]:
                            plot_routes(data, manager, routing, solution)

                        with solution_tabs[2]:
                            st.subheader("Solution Statistics")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Total Vehicles Used", data["num_vehicles"])

                            with col2:
                                total_distance = calculate_total_distance(routing, solution, manager, data)
                                st.metric("Total Distance (km)", f"{total_distance / 1000:.2f}")

                            with col3:
                                st.metric("Total Deliveries", total_demand)

                            # Additional statistics
                            st.write("---")
                            st.write("Detailed Statistics:")
                            st.write(f"- Total pickups: {total_pickup}")
                            st.write(f"- Available capacity: {total_capacity}")
                            st.write(
                                f"- Capacity utilization: {((total_demand + total_pickup) / total_capacity) * 100:.1f}%")

                    else:
                        st.error("""
                        ❌ No solution found! Try:
                        1. Adding more vehicles
                        2. Increasing vehicle capacities
                        3. Reducing customer demands
                        4. Adjusting pickup quantities
                        5. Increasing maximum distance limits
                        """)

                        # Display constraints for debugging
                        st.subheader("Current Constraints")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Demand and Capacity:")
                            st.write(f"- Total demand: {total_demand}")
                            st.write(f"- Total pickups: {total_pickup}")
                            st.write(f"- Total capacity: {total_capacity}")
                        with col2:
                            st.write("Vehicle Information:")
                            st.write(f"- Number of vehicles: {data['num_vehicles']}")
                            st.write(f"- Max distance: {max(data['vehicle_max_distances']) / 1000:.2f} km")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.write("Please check your input data and try again.")


if __name__ == "__main__":
    main()