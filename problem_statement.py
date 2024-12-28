import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_example_scenario():
    """Creates an example scenario with depot and customers"""
    # Create example data
    depot = (40.7128, -74.0060)  # NYC coordinates
    np.random.seed(42)
    n_customers = 10

    # Generate customer locations around depot
    customers = {
        'Customer': range(1, n_customers + 1),
        'Latitude': depot[0] + np.random.normal(0, 0.1, n_customers),
        'Longitude': depot[1] + np.random.normal(0, 0.1, n_customers),
        'Demand': np.random.randint(100, 1000, n_customers),
        'Time Window': [f"{h:02d}:00-{h + 2:02d}:00"
                        for h in np.random.randint(8, 17, n_customers)]
    }
    return depot, pd.DataFrame(customers)


def plot_example_scenario(depot, customers):
    """Plots the example scenario"""
    plt.figure(figsize=(10, 6))

    # Plot customers
    plt.scatter(customers['Longitude'], customers['Latitude'],
                s=customers['Demand'] / 10, alpha=0.6,
                c='blue', label='Customers')

    # Plot depot
    plt.scatter(depot[1], depot[0],
                color='red', marker='*', s=300,
                label='Depot')

    plt.title('Example VRP Scenario')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()

    return plt


def main():
    # Page configuration
    st.set_page_config(
        page_title="Vehicle Routing Problem Explanation",
        page_icon="🚚",
        layout="wide"
    )

    # Custom CSS for component styling
    st.markdown("""
        <style>
        .component-card {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            color: white;
        }
        .component-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .component-description {
            font-size: 1rem;
            opacity: 0.9;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and Introduction
    st.title("🚚 Understanding the Vehicle Routing Problem (VRP)")

    st.markdown("""
    <div style='background-color: #2C3E50; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        The Vehicle Routing Problem (VRP) is a complex optimization challenge in logistics and 
        transportation. It involves finding the optimal set of routes for a fleet of vehicles 
        to deliver goods or services to a set of customers.
    </div>
    """, unsafe_allow_html=True)

    # Problem Statement Section
    st.header("📋 Problem Statement")

    st.write("""
    Given:
    1. A central depot
    2. A set of customers with demands
    3. A fleet of vehicles with capacities

    Find:
    The optimal set of routes that minimizes total transportation cost while satisfying all constraints.
    """)

    # Interactive Example
    st.header("🎯 Interactive Example")

    depot, customers = create_example_scenario()

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = plot_example_scenario(depot, customers)
        st.pyplot(fig)

    with col2:
        st.write("### Example Data")
        st.dataframe(customers)

    # Key Components with improved styling
    st.header("🔑 Key Components")

    components = {
        "Depot": {
            "description": "Central location where vehicles start and end their routes",
            "color": "#FF6B6B"  # Coral Red
        },
        "Customers": {
            "description": "Locations that need to be served, each with specific demands",
            "color": "#4ECDC4"  # Turquoise
        },
        "Vehicles": {
            "description": "Fleet of vehicles with defined capacities and constraints",
            "color": "#45B7D1"  # Blue
        },
        "Routes": {
            "description": "Sequences of customers visited by each vehicle",
            "color": "#96CEB4"  # Sage Green
        },
        "Constraints": {
            "description": "Rules and limitations that must be satisfied",
            "color": "#D4A5A5"  # Dusty Rose
        },
        "Objective": {
            "description": "Goal to minimize total cost (usually distance or time)",
            "color": "#9B5DE5"  # Purple
        }
    }

    col1, col2 = st.columns(2)

    for i, (component, details) in enumerate(components.items()):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
                <div style='background-color: {details["color"]}; 
                           padding: 1rem; 
                           border-radius: 0.5rem; 
                           margin-bottom: 1rem;
                           color: white;
                           box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;'>
                        {component}
                    </div>
                    <div style='font-size: 1rem; opacity: 0.9;'>
                        {details["description"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Rest of the sections remain the same...
    # [Previous code for Constraints Section, Variants Section, etc.]

    # Constraints Section
    st.header("⚠️ Constraints")

    constraints_tabs = st.tabs([
        "Vehicle Constraints",
        "Customer Constraints",
        "Route Constraints"
    ])

    with constraints_tabs[0]:
        st.markdown("""
        ### Vehicle Constraints
        - Maximum capacity
        - Working hours
        - Speed limitations
        - Vehicle-customer compatibility
        """)

    with constraints_tabs[1]:
        st.markdown("""
        ### Customer Constraints
        - Time windows for delivery
        - Service time requirements
        - Special handling needs
        - Priority levels
        """)

    with constraints_tabs[2]:
        st.markdown("""
        ### Route Constraints
        - Start and end at depot
        - Maximum route duration
        - Break requirements
        - Road restrictions
        """)

    # Variants Section
    st.header("🔄 VRP Variants")

    variants = {
        "CVRP (Capacitated VRP)": [
            "Vehicles have limited capacity",
            "Must not exceed capacity while serving customers",
            "Focuses on weight or volume constraints"
        ],
        "VRPTW (VRP with Time Windows)": [
            "Customers specify time windows for delivery",
            "Must arrive within specified time ranges",
            "Includes service time at each location"
        ],
        "MDVRP (Multi-Depot VRP)": [
            "Multiple depot locations",
            "Vehicles can start from different depots",
            "Requires depot-customer assignment"
        ],
        "VRPPD (VRP with Pickup and Delivery)": [
            "Both pickup and delivery services",
            "Load changes during route",
            "Precedence constraints between locations"
        ]
    }

    for variant, details in variants.items():
        with st.expander(variant):
            for detail in details:
                st.write(f"- {detail}")

    # Business Impact
    st.header("💼 Business Impact")

    impact_cols = st.columns(3)

    with impact_cols[0]:
        st.metric(
            "Cost Reduction",
            "15-30%",
            "Typical savings range",
            help="Potential cost savings from optimal routing"
        )

    with impact_cols[1]:
        st.metric(
            "Service Level",
            "95%+",
            "Customer satisfaction",
            help="Improved on-time delivery performance"
        )

    with impact_cols[2]:
        st.metric(
            "Fleet Utilization",
            "25%↑",
            "Efficiency increase",
            help="Better use of vehicle capacity"
        )

    # Final Notes
    st.markdown("""
    ---
    ### 📚 Additional Resources

    For more information about VRP:
    - [Google OR-Tools Documentation](https://developers.google.com/optimization)
    - [VRP Research Papers](https://www.sciencedirect.com/topics/engineering/vehicle-routing-problem)
    - [Online VRP Solvers](https://github.com/topics/vehicle-routing-problem)
    """)


if __name__ == "__main__":
    main()
