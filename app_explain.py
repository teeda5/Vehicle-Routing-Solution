import streamlit as st


def create_process_section(title, description, code=None, details=None):
    """Helper function to create consistent process sections"""
    st.header(title)
    st.write(description)

    if code or details:
        with st.expander("🔍 View Details"):
            if code:
                st.code(code, language='python')
            if details:
                st.write(details)


def main():
    # Page Configuration
    st.set_page_config(
        page_title="VRP Optimization Process",
        page_icon="🚚",
        layout="wide"
    )

    # Title and Introduction
    st.title("🚚 Understanding VRP Route Optimization Process")
    st.markdown("""
    <div style='background-color: #2C3E50; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        This interactive guide explains what happens when you click the 'Optimize Routes' button 
        in the Vehicle Routing Problem solver. Follow each step to understand the complete process.
    </div>
    """, unsafe_allow_html=True)

    # Step 1: Data Model Creation
    create_process_section(
        "1️⃣ Data Model Creation",
        """
        The system begins by creating a comprehensive data model that includes:
        - 📍 Depot and customer locations
        - 📦 Demand and pickup quantities
        - 🚛 Vehicle fleet configuration
        - 📏 Distance calculations
        """,
        code='''
# Create data model with all necessary components
data = create_data_model(
    distance_type,
    st.session_state.demands,
    st.session_state.pickups
)

# Validate total load against vehicle capacities
total_demand = sum(data['demands'])
total_pickup = sum(data['pickups'])
total_capacity = sum(data['vehicle_capacities'])

if total_demand + total_pickup > total_capacity:
    raise ValueError("Total load exceeds fleet capacity")
        ''',
        details="""
        The data model contains:
        - Complete distance matrix
        - Vehicle specifications
        - Customer requirements
        - Capacity constraints
        """
    )

    # Step 2: Routing Model Setup
    create_process_section(
        "2️⃣ Routing Model Setup",
        """
        The routing model is initialized with:
        - 🎯 Index management for locations
        - 🔄 Callback functions setup
        - 📊 Distance and capacity dimensions
        """,
        code='''
# Initialize routing components
manager = pywrapcp.RoutingIndexManager(
    len(data["distance_matrix"]),
    data["num_vehicles"],
    data["depot"]
)
routing = pywrapcp.RoutingModel(manager)

# Register callback functions
transit_callback_index = routing.RegisterTransitCallback(distance_callback)
demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

# Add dimensions
routing.AddDimension(
    transit_callback_index,
    0,  # no slack
    max(data["vehicle_max_distances"]),
    True,  # start cumul to zero
    'Distance'
)
        '''
    )

    # Step 3: Constraint Setup
    create_process_section(
        "3️⃣ Constraint Configuration",
        """
        The system configures various constraints:

        1. **Vehicle Constraints**:
           - ⛽ Maximum travel distance
           - 📦 Loading capacity

        2. **Route Constraints**:
           - 🏠 Depot start/end points
           - 🔄 Pickup and delivery rules
        """
    )

    # Step 4: Solution Search
    create_process_section(
        "4️⃣ Solution Search Process",
        """
        The solver employs sophisticated search strategies:

        1. **Initial Solution**:
           - 🎯 PATH_CHEAPEST_ARC strategy
           - 🔍 Nearest neighbor consideration

        2. **Solution Improvement**:
           - 🔄 GUIDED_LOCAL_SEARCH
           - ⚡ Local optimization
           - 🎯 Global improvement
        """,
        code='''
# Configure search parameters
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
        '''
    )

    # Step 5: Solution Processing
    create_process_section(
        "5️⃣ Solution Processing",
        """
        When a solution is found, the system:

        1. **Analyzes Routes**:
           - 📊 Calculate statistics
           - 🗺️ Generate visualizations
           - 📈 Compute metrics

        2. **Presents Results**:
           - 🎯 Route details
           - 📊 Performance metrics
           - 🚛 Vehicle utilization
        """
    )

    # Error Handling Section
    st.header("⚠️ Error Handling")

    col1, col2 = st.columns(2)
    with col1:
        st.error("**Common Issues**")
        st.write("""
        - No feasible solution found
        - Capacity constraints violated
        - Distance limits exceeded
        - Insufficient vehicles
        """)

    with col2:
        st.success("**Solutions**")
        st.write("""
        - Add more vehicles
        - Increase capacities
        - Adjust demand quantities
        - Modify distance limits
        """)

    # Performance Metrics
    st.header("📊 Performance Considerations")

    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric(
            "Time Limit",
            "30 seconds",
            help="Maximum computation time allowed"
        )
    with metrics_cols[1]:
        st.metric(
            "Solution Limit",
            "100",
            help="Maximum number of solutions to consider"
        )
    with metrics_cols[2]:
        st.metric(
            "Optimization Goal",
            "Minimize Distance",
            help="Primary optimization objective"
        )

    # Final Notes
    st.markdown("""
    ---
    ### 💡 Technical Implementation

    This optimization process leverages:
    - **Google OR-Tools**: Advanced routing algorithms
    - **Constraint Programming**: For handling complex constraints
    - **Metaheuristics**: For solution improvement
    - **Mixed Integer Programming**: For optimal solution finding

    The implementation ensures efficient and effective route optimization while 
    maintaining all specified constraints and requirements.
    """)


if __name__ == "__main__":
    main()
