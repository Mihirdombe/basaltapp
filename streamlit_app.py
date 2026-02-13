"""
Basalt-PCM Thermal Buffer Sizing Calculator
A Streamlit app for calculating PCM requirements for thermal buffering in walls.

This application helps size phase change material (PCM) requirements for school-scale
thermal buffering experiments, using a simplified energy balance approach.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO


# ============================================================================
# UNIT CONVERSION FUNCTIONS
# ============================================================================

def ft2_to_m2(area_ft2):
    """Convert square feet to square meters."""
    return area_ft2 * 0.09290304


def m2_to_ft2(area_m2):
    """Convert square meters to square feet."""
    return area_m2 / 0.09290304


def kg_to_lb(mass_kg):
    """Convert kilograms to pounds."""
    return mass_kg * 2.20462262


def lb_to_kg(mass_lb):
    """Convert pounds to kilograms."""
    return mass_lb / 2.20462262


def btu_hr_ft2_f_to_w_m2_k(u_imperial):
    """Convert U-value from BTU/(hr·ft²·°F) to W/(m²·K)."""
    return u_imperial * 5.678263


def w_m2_k_to_btu_hr_ft2_f(u_metric):
    """Convert U-value from W/(m²·K) to BTU/(hr·ft²·°F)."""
    return u_metric / 5.678263


def celsius_to_fahrenheit_delta(delta_c):
    """Convert temperature difference from Celsius to Fahrenheit."""
    return delta_c * 1.8


def fahrenheit_to_celsius_delta(delta_f):
    """Convert temperature difference from Fahrenheit to Celsius."""
    return delta_f / 1.8


def liters_to_cubic_inches(liters):
    """Convert liters to cubic inches."""
    return liters * 61.0237


def m3_to_liters(volume_m3):
    """Convert cubic meters to liters."""
    return volume_m3 * 1000


def mm_to_inches(mm):
    """Convert millimeters to inches."""
    return mm / 25.4


# ============================================================================
# CORE CALCULATION FUNCTIONS
# ============================================================================

def compute_energy_per_area(u_value_w_m2_k, delta_t_k, duration_hours):
    """
    Calculate the energy per unit area that needs to be buffered.
    
    Parameters:
    -----------
    u_value_w_m2_k : float
        Overall heat transfer coefficient in W/(m²·K)
    delta_t_k : float
        Temperature difference in Kelvin (or Celsius difference)
    duration_hours : float
        Duration of buffering in hours
    
    Returns:
    --------
    float
        Energy per unit area in J/m²
    """
    duration_seconds = duration_hours * 3600
    heat_flux = u_value_w_m2_k * delta_t_k  # W/m²
    energy_per_area = heat_flux * duration_seconds  # J/m²
    return energy_per_area


def compute_effective_energy_per_kg(latent_heat_kj_kg, cp_kj_kg_k, usable_temp_swing_k):
    """
    Calculate the effective energy storage capacity of PCM per kilogram.
    
    Parameters:
    -----------
    latent_heat_kj_kg : float
        Latent heat of fusion in kJ/kg
    cp_kj_kg_k : float
        Specific heat capacity of PCM in kJ/(kg·K)
    usable_temp_swing_k : float
        Usable temperature swing around melting point in K
    
    Returns:
    --------
    float
        Effective energy per kg in kJ/kg
    """
    effective_energy = latent_heat_kj_kg + (cp_kj_kg_k * usable_temp_swing_k)
    return effective_energy


def compute_pcm_mass_per_area(energy_per_area_j_m2, effective_energy_kj_kg, safety_factor):
    """
    Calculate the required PCM mass per unit area.
    
    Parameters:
    -----------
    energy_per_area_j_m2 : float
        Energy to be buffered per unit area in J/m²
    effective_energy_kj_kg : float
        Effective energy storage capacity in kJ/kg
    safety_factor : float
        Safety factor multiplier (typically 1.0-2.0)
    
    Returns:
    --------
    float
        PCM mass per unit area in kg/m²
    """
    energy_per_area_kj_m2 = energy_per_area_j_m2 / 1000
    pcm_mass_per_area = (energy_per_area_kj_m2 / effective_energy_kj_kg) * safety_factor
    return pcm_mass_per_area


def compute_pcm_volume_and_thickness(pcm_mass_kg, pcm_density_kg_m3, wall_area_m2):
    """
    Calculate PCM volume and equivalent uniform thickness.
    
    Parameters:
    -----------
    pcm_mass_kg : float
        Total PCM mass in kg
    pcm_density_kg_m3 : float
        PCM density in kg/m³
    wall_area_m2 : float
        Wall area in m²
    
    Returns:
    --------
    tuple
        (volume_m3, thickness_m, volume_liters, thickness_mm)
    """
    volume_m3 = pcm_mass_kg / pcm_density_kg_m3
    thickness_m = volume_m3 / wall_area_m2
    volume_liters = m3_to_liters(volume_m3)
    thickness_mm = thickness_m * 1000
    return volume_m3, thickness_m, volume_liters, thickness_mm


def compute_composite_requirements(pcm_mass_kg, pcm_mass_fraction_percent, composite_density_kg_m3=None):
    """
    Calculate composite material requirements.
    
    Parameters:
    -----------
    pcm_mass_kg : float
        Required PCM mass in kg
    pcm_mass_fraction_percent : float
        PCM mass fraction in composite (percent)
    composite_density_kg_m3 : float, optional
        Composite density in kg/m³
    
    Returns:
    --------
    dict
        Dictionary containing composite mass and volume (if density provided)
    """
    pcm_fraction = pcm_mass_fraction_percent / 100
    total_composite_mass_kg = pcm_mass_kg / pcm_fraction
    
    result = {
        'composite_mass_kg': total_composite_mass_kg,
        'composite_mass_lb': kg_to_lb(total_composite_mass_kg)
    }
    
    if composite_density_kg_m3:
        composite_volume_m3 = total_composite_mass_kg / composite_density_kg_m3
        result['composite_volume_m3'] = composite_volume_m3
        result['composite_volume_liters'] = m3_to_liters(composite_volume_m3)
    
    return result


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_inputs(u_value, delta_t, duration, latent_heat, cp, usable_swing, density, safety_factor):
    """
    Validate user inputs and return warnings if any.
    
    Returns:
    --------
    list
        List of warning messages
    """
    warnings = []
    
    if u_value <= 0:
        warnings.append("U-value must be positive.")
    if delta_t <= 0.1:
        warnings.append("Temperature difference is very small. Results may be near zero.")
    if duration <= 0:
        warnings.append("Duration must be positive.")
    if latent_heat <= 0:
        warnings.append("Latent heat must be positive.")
    if cp <= 0:
        warnings.append("Specific heat capacity must be positive.")
    if usable_swing <= 0:
        warnings.append("Usable temperature swing must be positive.")
    if usable_swing > 20:
        warnings.append("Warning: Usable temperature swing > 20K is unusually large for typical PCMs.")
    if density <= 0:
        warnings.append("PCM density must be positive.")
    if safety_factor < 1.0:
        warnings.append("Safety factor should typically be >= 1.0.")
    
    return warnings


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_sensitivity_plot_duration(u_value_w_m2_k, delta_t_k, latent_heat, cp, usable_swing, 
                                     safety_factor, wall_area_ft2, duration_range=None):
    """
    Create sensitivity plot of PCM requirement vs. duration.
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    if duration_range is None:
        duration_range = np.linspace(1, 24, 50)
    
    pcm_lb_ft2_values = []
    
    for duration in duration_range:
        energy_per_area = compute_energy_per_area(u_value_w_m2_k, delta_t_k, duration)
        effective_energy = compute_effective_energy_per_kg(latent_heat, cp, usable_swing)
        pcm_mass_per_area_kg_m2 = compute_pcm_mass_per_area(energy_per_area, effective_energy, safety_factor)
        
        # Convert to lb/ft²
        pcm_mass_per_area_lb_ft2 = kg_to_lb(pcm_mass_per_area_kg_m2) / m2_to_ft2(1)
        pcm_lb_ft2_values.append(pcm_mass_per_area_lb_ft2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(duration_range, pcm_lb_ft2_values, linewidth=2, color='#1f77b4')
    ax.set_xlabel('Duration (hours)', fontsize=12)
    ax.set_ylabel('PCM Required (lb/ft²)', fontsize=12)
    ax.set_title('PCM Requirement vs. Buffering Duration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(duration_range[0], duration_range[-1])
    
    return fig


def create_sensitivity_plot_delta_t(u_value_w_m2_k, duration_hours, latent_heat, cp, usable_swing,
                                     safety_factor, wall_area_ft2, delta_t_range=None):
    """
    Create sensitivity plot of PCM requirement vs. temperature difference.
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    if delta_t_range is None:
        delta_t_range = np.linspace(1, 30, 50)
    
    pcm_lb_ft2_values = []
    
    for delta_t in delta_t_range:
        energy_per_area = compute_energy_per_area(u_value_w_m2_k, delta_t, duration_hours)
        effective_energy = compute_effective_energy_per_kg(latent_heat, cp, usable_swing)
        pcm_mass_per_area_kg_m2 = compute_pcm_mass_per_area(energy_per_area, effective_energy, safety_factor)
        
        # Convert to lb/ft²
        pcm_mass_per_area_lb_ft2 = kg_to_lb(pcm_mass_per_area_kg_m2) / m2_to_ft2(1)
        pcm_lb_ft2_values.append(pcm_mass_per_area_lb_ft2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(delta_t_range, pcm_lb_ft2_values, linewidth=2, color='#ff7f0e')
    ax.set_xlabel('Temperature Difference (°C or K)', fontsize=12)
    ax.set_ylabel('PCM Required (lb/ft²)', fontsize=12)
    ax.set_title('PCM Requirement vs. Temperature Difference', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(delta_t_range[0], delta_t_range[-1])
    
    return fig


# ============================================================================
# CSV EXPORT FUNCTION
# ============================================================================

def generate_csv_summary(inputs, outputs):
    """
    Generate a CSV summary of inputs and outputs.
    
    Returns:
    --------
    str
        CSV formatted string
    """
    data = []
    
    # Add header
    data.append(['Basalt-PCM Thermal Buffer Sizing Calculator - Summary'])
    data.append([''])
    
    # Add inputs
    data.append(['INPUTS'])
    for key, value in inputs.items():
        data.append([key, value])
    data.append([''])
    
    # Add outputs
    data.append(['OUTPUTS'])
    for key, value in outputs.items():
        data.append([key, value])
    
    # Convert to CSV string
    output = StringIO()
    for row in data:
        output.write(','.join(map(str, row)) + '\n')
    
    return output.getvalue()


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    """Main function to run the Streamlit app."""
    
    # Page configuration
    st.set_page_config(
        page_title="PCM Thermal Buffer Calculator",
        page_icon="🔥",
        layout="wide"
    )
    
    # Title and description
    st.title("🔥 Basalt-PCM Thermal Buffer Sizing Calculator")
    st.markdown("""
    This application calculates the required Phase Change Material (PCM) mass needed to buffer 
    temperature fluctuations in a wall assembly. The calculator uses a simplified energy balance 
    approach suitable for school-scale thermal buffering experiments.
    """)
    
    # Sidebar for inputs
    st.sidebar.header("📊 Input Parameters")
    
    # ========================================================================
    # SECTION A: Geometry and Design Target
    # ========================================================================
    st.sidebar.subheader("A. Geometry and Design Target")
    
    wall_area_ft2 = st.sidebar.number_input(
        "Wall Area (ft²)",
        min_value=0.1,
        value=100.0,
        step=1.0,
        help="Total wall area to be analyzed"
    )
    
    units_system = st.sidebar.radio(
        "Preferred Units",
        options=["Imperial", "Metric"],
        help="Primary unit system for inputs (outputs always include both)"
    )
    
    duration_hours = st.sidebar.number_input(
        "Duration of Buffering (hours)",
        min_value=0.1,
        value=8.0,
        step=0.5,
        help="Time period over which thermal buffering is required"
    )
    
    # Temperature difference input
    if units_system == "Imperial":
        delta_t_input = st.sidebar.number_input(
            "Temperature Difference (°F)",
            min_value=0.1,
            value=20.0,
            step=1.0,
            help="Temperature difference between outside and inside"
        )
        delta_t_k = fahrenheit_to_celsius_delta(delta_t_input)
    else:
        delta_t_input = st.sidebar.number_input(
            "Temperature Difference (°C)",
            min_value=0.1,
            value=11.1,
            step=0.5,
            help="Temperature difference between outside and inside"
        )
        delta_t_k = delta_t_input
    
    # U-value input
    u_value_units = st.sidebar.selectbox(
        "U-value Units",
        options=["W/(m²·K)", "BTU/(hr·ft²·°F)"],
        help="Select units for thermal transmittance input"
    )
    
    if u_value_units == "W/(m²·K)":
        u_value_input = st.sidebar.number_input(
            "U-value (W/(m²·K))",
            min_value=0.01,
            value=0.5,
            step=0.1,
            help="Overall heat transfer coefficient"
        )
        u_value_w_m2_k = u_value_input
    else:
        u_value_input = st.sidebar.number_input(
            "U-value (BTU/(hr·ft²·°F))",
            min_value=0.01,
            value=0.088,
            step=0.01,
            help="Overall heat transfer coefficient"
        )
        u_value_w_m2_k = btu_hr_ft2_f_to_w_m2_k(u_value_input)
    
    safety_factor = st.sidebar.slider(
        "Safety Factor",
        min_value=1.0,
        max_value=2.0,
        value=1.2,
        step=0.1,
        help="Multiplier to account for uncertainties and ensure adequate capacity"
    )
    
    # ========================================================================
    # SECTION B: PCM Properties
    # ========================================================================
    st.sidebar.subheader("B. PCM Properties")
    
    latent_heat_kj_kg = st.sidebar.number_input(
        "Latent Heat of Fusion (kJ/kg)",
        min_value=1.0,
        value=200.0,
        step=10.0,
        help="Latent heat of phase change for the PCM"
    )
    
    cp_pcm_kj_kg_k = st.sidebar.number_input(
        "Specific Heat Capacity (kJ/(kg·K))",
        min_value=0.1,
        value=2.0,
        step=0.1,
        help="Specific heat capacity of the PCM"
    )
    
    usable_temp_swing_k = st.sidebar.number_input(
        "Usable Temperature Swing (K)",
        min_value=0.1,
        value=6.0,
        step=0.5,
        help="Effective temperature range around melting point where PCM operates"
    )
    
    pcm_density_kg_m3 = st.sidebar.number_input(
        "PCM Density (kg/m³)",
        min_value=1.0,
        value=800.0,
        step=50.0,
        help="Density of the phase change material"
    )
    
    # ========================================================================
    # SECTION C: Optional Composite Constraints
    # ========================================================================
    st.sidebar.subheader("C. Optional: Composite Constraints")
    
    use_composite = st.sidebar.checkbox(
        "Calculate Composite Requirements",
        value=False,
        help="Enable to calculate total composite material needed"
    )
    
    if use_composite:
        pcm_mass_fraction = st.sidebar.number_input(
            "PCM Mass Fraction in Composite (%)",
            min_value=1.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            help="Percentage of PCM in the composite material by mass"
        )
        
        composite_density_kg_m3 = st.sidebar.number_input(
            "Composite Density (kg/m³)",
            min_value=1.0,
            value=1200.0,
            step=50.0,
            help="Density of the composite material"
        )
    
    # ========================================================================
    # OPTIONAL: Design Mode
    # ========================================================================
    st.sidebar.subheader("Optional: Direct Energy Input Mode")
    
    use_direct_energy = st.sidebar.checkbox(
        "Use Direct Energy Input",
        value=False,
        help="Specify buffered energy directly instead of calculating from U, deltaT, and duration"
    )
    
    if use_direct_energy:
        target_energy_wh_ft2 = st.sidebar.number_input(
            "Target Energy per Area (Wh/ft²)",
            min_value=0.1,
            value=100.0,
            step=10.0,
            help="Directly specify the energy to be buffered per square foot"
        )
    
    # ========================================================================
    # PERFORM CALCULATIONS
    # ========================================================================
    
    # Convert wall area to m²
    wall_area_m2 = ft2_to_m2(wall_area_ft2)
    
    # Validate inputs
    warnings = validate_inputs(
        u_value_w_m2_k, delta_t_k, duration_hours,
        latent_heat_kj_kg, cp_pcm_kj_kg_k, usable_temp_swing_k,
        pcm_density_kg_m3, safety_factor
    )
    
    # Display warnings if any
    if warnings:
        st.sidebar.warning("⚠️ Input Validation Warnings:")
        for warning in warnings:
            st.sidebar.warning(f"• {warning}")
    
    # Calculate energy per area
    if use_direct_energy:
        # Convert Wh/ft² to J/m²
        energy_per_area_wh_m2 = target_energy_wh_ft2 * m2_to_ft2(1)
        energy_per_area_j_m2 = energy_per_area_wh_m2 * 3600  # Wh to J
    else:
        energy_per_area_j_m2 = compute_energy_per_area(u_value_w_m2_k, delta_t_k, duration_hours)
    
    # Calculate effective energy per kg
    effective_energy_kj_kg = compute_effective_energy_per_kg(
        latent_heat_kj_kg, cp_pcm_kj_kg_k, usable_temp_swing_k
    )
    
    # Calculate PCM mass per area
    pcm_mass_per_area_kg_m2 = compute_pcm_mass_per_area(
        energy_per_area_j_m2, effective_energy_kj_kg, safety_factor
    )
    
    # Convert to lb/ft²
    pcm_mass_per_area_lb_ft2 = kg_to_lb(pcm_mass_per_area_kg_m2) / m2_to_ft2(1)
    
    # Calculate total PCM mass
    total_pcm_kg = pcm_mass_per_area_kg_m2 * wall_area_m2
    total_pcm_lb = kg_to_lb(total_pcm_kg)
    
    # Calculate volume and thickness
    volume_m3, thickness_m, volume_liters, thickness_mm = compute_pcm_volume_and_thickness(
        total_pcm_kg, pcm_density_kg_m3, wall_area_m2
    )
    
    volume_cubic_inches = liters_to_cubic_inches(volume_liters)
    thickness_inches = mm_to_inches(thickness_mm)
    
    # ========================================================================
    # DISPLAY MAIN OUTPUTS
    # ========================================================================
    
    st.header("📈 Calculation Results")
    
    # Create columns for main outputs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="PCM Required",
            value=f"{pcm_mass_per_area_lb_ft2:.3f} lb/ft²",
            help="Primary output: PCM mass per unit area in imperial units"
        )
        st.caption(f"({pcm_mass_per_area_kg_m2:.3f} kg/m²)")
    
    with col2:
        st.metric(
            label="Total PCM Mass",
            value=f"{total_pcm_lb:.1f} lb",
            help="Total PCM mass required for the entire wall"
        )
        st.caption(f"({total_pcm_kg:.1f} kg)")
    
    with col3:
        st.metric(
            label="PCM Volume",
            value=f"{volume_liters:.2f} L",
            help="Total volume of PCM required"
        )
        st.caption(f"({volume_cubic_inches:.1f} in³)")
    
    with col4:
        st.metric(
            label="Equivalent PCM Thickness",
            value=f"{thickness_mm:.2f} mm",
            help="Thickness if PCM is spread uniformly over wall area"
        )
        st.caption(f"({thickness_inches:.3f} in)")
    
    # ========================================================================
    # COMPOSITE REQUIREMENTS (if enabled)
    # ========================================================================
    
    if use_composite:
        st.subheader("🧱 Composite Material Requirements")
        
        composite_results = compute_composite_requirements(
            total_pcm_kg, pcm_mass_fraction, composite_density_kg_m3
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Total Composite Mass",
                value=f"{composite_results['composite_mass_lb']:.1f} lb",
                help=f"Total composite mass needed to achieve {pcm_mass_fraction}% PCM fraction"
            )
            st.caption(f"({composite_results['composite_mass_kg']:.1f} kg)")
        
        with col2:
            if 'composite_volume_liters' in composite_results:
                st.metric(
                    label="Composite Volume",
                    value=f"{composite_results['composite_volume_liters']:.2f} L",
                    help="Total volume of composite material"
                )
                st.caption(f"({composite_results['composite_volume_m3']:.3f} m³)")
        
        composite_mass_per_area_kg_m2 = composite_results['composite_mass_kg'] / wall_area_m2
        composite_mass_per_area_lb_ft2 = composite_results['composite_mass_lb'] / wall_area_ft2
        
        st.info(f"""
        **Composite per unit area:** {composite_mass_per_area_lb_ft2:.3f} lb/ft² 
        ({composite_mass_per_area_kg_m2:.3f} kg/m²)
        """)
    
    # ========================================================================
    # EXPLANATION AND FORMULAS
    # ========================================================================
    
    st.header("📚 Model Explanation")
    
    st.markdown("""
    This calculator uses a simplified energy balance model to estimate PCM requirements for thermal buffering.
    The approach assumes that the PCM absorbs and releases heat primarily through latent heat near its melting point,
    which slows temperature changes in the wall assembly.
    """)
    
    with st.expander("View Detailed Formulas and Assumptions", expanded=False):
        st.markdown("""
        ### Calculation Steps:
        
        **Step 1: Calculate Energy to be Buffered**
        
        The energy per unit area that must be buffered over the specified duration is calculated as:
        
        Energy per Area (J/m²) = U-value (W/(m²·K)) × Temperature Difference (K) × Duration (seconds)
        
        Where Duration (seconds) = Duration (hours) × 3600
        
        **Step 2: Calculate Effective PCM Energy Storage Capacity**
        
        The effective energy storage capacity of the PCM combines both latent heat and sensible heat:
        
        Effective Energy (kJ/kg) = Latent Heat (kJ/kg) + [Specific Heat (kJ/(kg·K)) × Usable Temp Swing (K)]
        
        **Step 3: Calculate Required PCM Mass**
        
        The PCM mass per unit area is calculated by dividing the energy to be buffered by the effective storage capacity:
        
        PCM Mass per Area (kg/m²) = [Energy per Area (kJ/m²) / Effective Energy (kJ/kg)] × Safety Factor
        
        Note: Energy per Area is converted from J/m² to kJ/m² by dividing by 1000.
        
        **Step 4: Calculate Total Mass, Volume, and Thickness**
        
        - Total PCM Mass (kg) = PCM Mass per Area (kg/m²) × Wall Area (m²)
        - PCM Volume (m³) = Total PCM Mass (kg) / PCM Density (kg/m³)
        - Equivalent Thickness (m) = PCM Volume (m³) / Wall Area (m²)
        
        ### Key Assumptions:
        
        1. **Steady-state temperature difference:** The model assumes a constant deltaT over the buffering period.
        2. **Lumped energy buffering:** Heat storage is treated as a lumped parameter without detailed spatial distribution.
        3. **No solar gain variations:** Solar radiation effects are not explicitly modeled.
        4. **Uniform PCM distribution:** Calculations assume PCM is uniformly distributed across the wall.
        5. **Complete phase change utilization:** The model assumes the PCM operates within its optimal phase change range.
        6. **One-dimensional heat transfer:** Heat flow is assumed perpendicular to the wall surface.
        
        ### Unit Conversions Applied:
        
        - 1 ft² = 0.09290304 m²
        - 1 lb = 0.453592 kg
        - 1 BTU/(hr·ft²·°F) = 5.678263 W/(m²·K)
        - Temperature difference: ΔT(°F) = ΔT(°C) × 1.8
        - 1 liter = 61.0237 cubic inches
        - 1 mm = 0.03937 inches
        """)
    
    # ========================================================================
    # SENSITIVITY PLOTS
    # ========================================================================
    
    st.header("📊 Sensitivity Analysis")
    
    st.markdown("""
    The following charts show how the PCM requirement changes with different parameters,
    helping you understand the sensitivity of the design to key variables.
    """)
    
    # Plot 1: Duration sensitivity
    st.subheader("PCM Requirement vs. Buffering Duration")
    fig1 = create_sensitivity_plot_duration(
        u_value_w_m2_k, delta_t_k, latent_heat_kj_kg,
        cp_pcm_kj_kg_k, usable_temp_swing_k, safety_factor, wall_area_ft2
    )
    st.pyplot(fig1)
    plt.close(fig1)
    
    st.caption(f"""
    This plot shows how PCM requirements scale with the duration of buffering, 
    assuming U-value = {u_value_w_m2_k:.2f} W/(m²·K) and ΔT = {delta_t_k:.2f} K.
    """)
    
    # Plot 2: Temperature difference sensitivity
    st.subheader("PCM Requirement vs. Temperature Difference")
    fig2 = create_sensitivity_plot_delta_t(
        u_value_w_m2_k, duration_hours, latent_heat_kj_kg,
        cp_pcm_kj_kg_k, usable_temp_swing_k, safety_factor, wall_area_ft2
    )
    st.pyplot(fig2)
    plt.close(fig2)
    
    st.caption(f"""
    This plot shows how PCM requirements scale with temperature difference, 
    assuming U-value = {u_value_w_m2_k:.2f} W/(m²·K) and Duration = {duration_hours:.1f} hours.
    """)
    
    # ========================================================================
    # DOWNLOAD SUMMARY
    # ========================================================================
    
    st.header("💾 Export Results")
    
    # Prepare inputs and outputs for CSV
    inputs_dict = {
        'Wall Area (ft²)': wall_area_ft2,
        'Wall Area (m²)': wall_area_m2,
        'Duration (hours)': duration_hours,
        'Temperature Difference (K)': delta_t_k,
        'U-value (W/(m²·K))': u_value_w_m2_k,
        'Safety Factor': safety_factor,
        'Latent Heat (kJ/kg)': latent_heat_kj_kg,
        'Specific Heat (kJ/(kg·K))': cp_pcm_kj_kg_k,
        'Usable Temp Swing (K)': usable_temp_swing_k,
        'PCM Density (kg/m³)': pcm_density_kg_m3
    }
    
    outputs_dict = {
        'PCM Required (lb/ft²)': f"{pcm_mass_per_area_lb_ft2:.4f}",
        'PCM Required (kg/m²)': f"{pcm_mass_per_area_kg_m2:.4f}",
        'Total PCM (lb)': f"{total_pcm_lb:.2f}",
        'Total PCM (kg)': f"{total_pcm_kg:.2f}",
        'PCM Volume (liters)': f"{volume_liters:.3f}",
        'PCM Volume (cubic inches)': f"{volume_cubic_inches:.2f}",
        'Equivalent Thickness (mm)': f"{thickness_mm:.3f}",
        'Equivalent Thickness (inches)': f"{thickness_inches:.4f}",
        'Energy per Area (kJ/m²)': f"{energy_per_area_j_m2/1000:.2f}",
        'Effective Energy per kg (kJ/kg)': f"{effective_energy_kj_kg:.2f}"
    }
    
    if use_composite:
        inputs_dict['PCM Mass Fraction (%)'] = pcm_mass_fraction
        inputs_dict['Composite Density (kg/m³)'] = composite_density_kg_m3
        outputs_dict['Composite Mass (lb)'] = f"{composite_results['composite_mass_lb']:.2f}"
        outputs_dict['Composite Mass (kg)'] = f"{composite_results['composite_mass_kg']:.2f}"
    
    csv_data = generate_csv_summary(inputs_dict, outputs_dict)
    
    st.download_button(
        label="📥 Download Summary as CSV",
        data=csv_data,
        file_name="pcm_calculator_summary.csv",
        mime="text/csv",
        help="Download a CSV file containing all inputs and outputs"
    )
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>Basalt-PCM Thermal Buffer Sizing Calculator</strong></p>
    <p>For school-scale thermal buffering experiments | Version 1.0</p>
    <p><em>Note: This calculator provides estimates based on simplified assumptions. 
    For critical applications, consult with thermal engineering professionals.</em></p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
