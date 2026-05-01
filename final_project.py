"""
Name:       Student Name
CS230:      Section XXX
Data:       Airports Around the World (airport-codes.csv)
URL:        (Streamlit Cloud link after deployment)

Description:
    This program explores a dataset of over 85,000 airports from around the
    world. It allows users to filter airports by type, country, and continent,
    then visualizes the results through an interactive PyDeck map, a bar chart
    comparing airport counts by country, and a heatmap showing airport type
    distribution across continents. Users can also examine elevation statistics,
    find the highest and lowest airports, and explore a sortable, filterable
    data table. The app uses a sidebar for navigation across three pages:
    Overview, Explore, and Statistics.

References:
    - Streamlit documentation: https://docs.streamlit.io
    - PyDeck documentation: https://deckgl.readthedocs.io
    - Pandas documentation: https://pandas.pydata.org/docs/
    - Matplotlib documentation: https://matplotlib.org/stable/contents.html
    - Seaborn documentation: https://seaborn.pydata.org
    - Dataset source: https://datahub.io/core/airport-codes
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pydeck as pdk

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="World Airport Explorer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# #[ST3] - Custom CSS for polished page design (colors, fonts, sidebar)
st.markdown(
    """
    <style>
    .main { background-color: #f5f7fa; }
    h1 { color: #1a3c5e; font-family: 'Arial', sans-serif; }
    h2, h3 { color: #2a5080; font-family: 'Arial', sans-serif; }
    .block-container { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] { background-color: #1a3c5e; }
    section[data-testid="stSidebar"] * { color: #e8f0fa !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTINENT_NAMES = {
    "AF": "Africa",
    "AN": "Antarctica",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "OC": "Oceania",
    "SA": "South America",
}

TYPE_COLORS = {
    "large_airport":  [30,  110, 220, 200],
    "medium_airport": [50,  190, 110, 180],
    "small_airport":  [230, 150,  40, 160],
    "heliport":       [170,  70, 210, 150],
    "seaplane_base":  [ 80, 210, 230, 160],
    "closed":         [140, 140, 140, 110],
    "balloonport":    [220,  60,  60, 160],
}

# ---------------------------------------------------------------------------
# Data loading (cached for performance)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load airport-codes.csv, parse the 'coordinates' column into separate
    latitude_deg and longitude_deg floats, add a continent_name column,
    and drop rows without usable positional data.

    Returns:
        pd.DataFrame: Cleaned airport dataset ready for mapping and analysis.
    """
    df = pd.read_csv("airport-codes.csv")

    # The dataset stores coordinates as a single "lat, lon" string column.
    # Split and convert to float so PyDeck and filters can use them.
    coords = df["coordinates"].str.split(",", expand=True)
    df["latitude_deg"]  = pd.to_numeric(coords[0].str.strip(), errors="coerce")
    df["longitude_deg"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")

    # #[COLUMNS] - Add human-readable continent name column; drop raw coordinates column
    df["continent_name"] = df["continent"].map(CONTINENT_NAMES)
    df = df.drop(columns=["coordinates"])

    # Drop rows missing lat/lon (cannot place on map)
    df = df.dropna(subset=["latitude_deg", "longitude_deg"])

    # Ensure elevation is numeric
    df["elevation_ft"] = pd.to_numeric(df["elevation_ft"], errors="coerce")

    return df


df = load_data()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
# NOTE: main() is defined after helpers and called at the bottom of this file.

# #[FUNC2P] - Two parameters; country has a default value
def filter_airports(dataframe, airport_types, country=None):
    """
    Filter the airport DataFrame by type list and an optional country code.

    Parameters:
        dataframe (pd.DataFrame): Full airport dataset.
        airport_types (list): List of airport type strings to include.
        country (str, optional): ISO country code to filter by. Default None = all.

    Returns:
        pd.DataFrame: Filtered subset.
    """
    # #[FILTER1] - Filter by a single condition: airport type
    result = dataframe[dataframe["type"].isin(airport_types)]

    # #[FILTER2] - Filter by type AND country (two conditions joined by AND)
    if country and country != "All":
        result = result[result["iso_country"] == country]

    return result


# #[FUNCRETURN2] - Returns two values
def get_elevation_extremes(dataframe):
    """
    Find the airports with the highest and lowest elevation_ft values.

    Parameters:
        dataframe (pd.DataFrame): Airport dataset containing elevation_ft.

    Returns:
        tuple: (highest_row, lowest_row) as pd.Series, or (None, None) if no data.
    """
    elev_df = dataframe.dropna(subset=["elevation_ft"])
    if elev_df.empty:
        return None, None
    # #[MAXMIN] - idxmax/idxmin to find largest and smallest elevation
    highest = elev_df.loc[elev_df["elevation_ft"].idxmax()]
    lowest  = elev_df.loc[elev_df["elevation_ft"].idxmin()]
    return highest, lowest


# #[FUNCCALL2] - Called once on Overview page, once on Statistics page
def count_by_group(dataframe, group_col):
    """
    Count airports per unique value in group_col, sorted descending.

    Parameters:
        dataframe (pd.DataFrame): Airport dataset.
        group_col (str): Column name to group by.

    Returns:
        pd.Series: Counts indexed by group values.
    """
    return dataframe.groupby(group_col)["ident"].count().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main function for the World Airport Explorer Streamlit app.
    Builds the sidebar controls, applies filters, and renders the selected page.
    """
    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    st.sidebar.markdown("## ✈️ World Airports")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Navigation**")

    # #[ST1] - Dropdown (selectbox) for page navigation
    page = st.sidebar.selectbox(
        "Go to page:",
        ["Overview", "Explore Airports", "Statistics"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")

    # #[ST1] - Multi-select for airport types (second selectbox/dropdown-style widget)
    all_types = sorted(df["type"].dropna().unique().tolist())
    selected_types = st.sidebar.multiselect(
        "Airport Type(s)",
        options=all_types,
        default=["large_airport", "medium_airport", "small_airport"],
    )

    # Continent selectbox
    continent_opts = ["All"] + sorted(
        [v for k, v in CONTINENT_NAMES.items() if k in df["continent"].dropna().unique()]
    )
    selected_continent_name = st.sidebar.selectbox("Continent", continent_opts)

    # Country selectbox (scoped to continent if one is chosen)
    if selected_continent_name != "All":
        cont_code_map = {v: k for k, v in CONTINENT_NAMES.items()}
        cont_code = cont_code_map.get(selected_continent_name)
        avail_countries = df[df["continent"] == cont_code]["iso_country"].dropna().unique().tolist()
    else:
        avail_countries = df["iso_country"].dropna().unique().tolist()

    selected_country = st.sidebar.selectbox("Country (ISO Code)", ["All"] + sorted(avail_countries))

    # Build the filtered DataFrame used across all pages
    filtered_df = filter_airports(df, selected_types, country=selected_country)

    if selected_continent_name != "All":
        cont_code_map = {v: k for k, v in CONTINENT_NAMES.items()}
        cont_code = cont_code_map.get(selected_continent_name)
        # Two-condition filter: type + country already applied; now add continent
        filtered_df = filtered_df[filtered_df["continent"] == cont_code]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing {len(filtered_df):,} airports**")

    # ---------------------------------------------------------------------------
    # PAGE 1: Overview
    # ---------------------------------------------------------------------------
    if page == "Overview":
        st.title("✈️ World Airport Explorer")
        st.markdown(
            "Explore a dataset of **85,000+ real-world airports** from "
            "[DataHub.io](https://datahub.io/core/airport-codes). "
            "Use the sidebar to filter by type, continent, and country, then "
            "browse the map and preview table below."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Airports (filtered)", f"{len(filtered_df):,}")
        c2.metric("Countries",           filtered_df["iso_country"].nunique())
        c3.metric("Continents",          filtered_df["continent"].nunique())
        c4.metric("Airport Types",       filtered_df["type"].nunique())

        st.markdown("---")

        # #[MAP] - PyDeck ScatterplotLayer: color by type, tooltip on hover
        st.subheader("🗺️ Airport Locations Map")
        st.markdown(
            "Each dot is color-coded by type: "
            "**blue** = large, **green** = medium, **orange** = small, "
            "**purple** = heliport, **teal** = seaplane base, **gray** = closed/other. "
            "Hover over any dot to see the name, location, type, and elevation."
        )

        map_df = filtered_df.copy()
        map_df["color"] = map_df["type"].apply(
            lambda t: TYPE_COLORS.get(t, [150, 150, 150, 130])
        )
        map_df["tooltip_text"] = (
            map_df["name"].fillna("Unknown")
            + "\n"
            + map_df["municipality"].fillna("").astype(str)
            + ", "
            + map_df["iso_country"].fillna("")
            + "\nType: "
            + map_df["type"].fillna("")
            + "\nElevation: "
            + map_df["elevation_ft"].fillna(0).astype(int).astype(str)
            + " ft"
        )

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["longitude_deg", "latitude_deg"],
            get_color="color",
            get_radius=8000,
            radius_min_pixels=2,
            radius_max_pixels=10,
            pickable=True,
            auto_highlight=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[scatter_layer],
                initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.3, pitch=0),
                tooltip={"text": "{tooltip_text}"},
                map_style="mapbox://styles/mapbox/light-v10",
            )
        )

        # Quick data preview, sorted by country then name  #[SORT]
        st.subheader("📋 Data Preview")
        preview_cols = ["ident", "iata_code", "name", "type",
                        "municipality", "iso_country", "continent_name", "elevation_ft"]
        sorted_preview = (
            filtered_df[preview_cols]
            .sort_values(["iso_country", "name"])
            .reset_index(drop=True)
        )
        st.dataframe(sorted_preview.head(50), width="stretch")

    # ---------------------------------------------------------------------------
    # PAGE 2: Explore Airports
    # ---------------------------------------------------------------------------
    elif page == "Explore Airports":
        st.title("🔍 Explore Airports")
        st.markdown(
            "Charts and a sortable table to dig into the filtered airports. "
            "Adjust the sidebar options to compare regions and types."
        )

        if filtered_df.empty:
            st.warning("No airports match the current filters. Adjust the sidebar options.")
            st.stop()

        # #[CHART1] - Bar chart: airports by country, custom colors and labels
        st.subheader("📊 Top Countries by Airport Count")
        st.markdown(
            "The 20 countries with the most airports in the current filtered selection. "
            "Bar color scales from light to dark with count."
        )

        # First call to count_by_group  #[FUNCCALL2]
        country_counts = count_by_group(filtered_df, "iso_country").head(20)
        norm = (country_counts.values - country_counts.values.min()) / (
            country_counts.values.max() - country_counts.values.min() + 1e-9
        )
        bar_colors = plt.cm.Blues(0.35 + 0.55 * norm)

        fig1, ax1 = plt.subplots(figsize=(13, 5))
        bars = ax1.bar(
            country_counts.index, country_counts.values,
            color=bar_colors, edgecolor="white", linewidth=0.7,
        )
        ax1.set_title("Top 20 Countries by Airport Count", fontsize=14, fontweight="bold", pad=12)
        ax1.set_xlabel("Country (ISO Code)", fontsize=11)
        ax1.set_ylabel("Number of Airports",  fontsize=11)
        ax1.tick_params(axis="x", rotation=45, labelsize=9)
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, val in zip(bars, country_counts.values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(country_counts.values) * 0.005,
                f"{val:,}", ha="center", va="bottom", fontsize=7.5, color="#333",
            )
        plt.tight_layout()
        st.pyplot(fig1)

        # #[CHART2] - Seaborn heatmap: airport type vs continent
        st.subheader("🌍 Airport Type Distribution by Continent")
        st.markdown(
            "A heatmap of airport type counts across continents. "
            "Darker cells indicate more airports of that type in that region."
        )

        heat_df = filtered_df.dropna(subset=["continent"]).copy()
        heat_df["continent_label"] = heat_df["continent"].map(CONTINENT_NAMES)
        pivot = (
            heat_df
            .groupby(["continent_label", "type"])["ident"]
            .count()
            .unstack(fill_value=0)
        )

        if not pivot.empty and pivot.shape[1] > 0:
            fig2, ax2 = plt.subplots(figsize=(11, max(3, len(pivot) * 0.75)))
            sns.heatmap(
                pivot,
                annot=True,
                fmt="d",
                cmap="YlOrBr",
                linewidths=0.5,
                ax=ax2,
                cbar_kws={"label": "Airport Count"},
            )
            ax2.set_title("Airport Types Across Continents", fontsize=13, fontweight="bold", pad=10)
            ax2.set_xlabel("Airport Type", fontsize=11)
            ax2.set_ylabel("Continent",    fontsize=11)
            ax2.tick_params(axis="x", rotation=30, labelsize=9)
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Not enough continent data for the heatmap with current filters.")

        # Sortable table
        st.subheader("📋 Filtered Airport Table")

        # #[ST2] - Slider to control how many rows to show
        max_rows = st.slider(
            "Rows to display",
            min_value=5,
            max_value=max(5, min(500, len(filtered_df))),
            value=min(25, max(5, len(filtered_df))),
            step=5,
        )

        sort_col = st.selectbox(
            "Sort table by:",
            ["name", "type", "iso_country", "continent_name", "elevation_ft", "municipality"],
        )
        sort_asc = st.radio("Order:", ["Ascending", "Descending"], horizontal=True) == "Ascending"

        show_cols = ["ident", "iata_code", "name", "type",
                     "municipality", "iso_country", "continent_name", "elevation_ft"]
        # #[SORT] - Sort the table by a user-selected column
        display_df = (
            filtered_df[show_cols]
            .sort_values(sort_col, ascending=sort_asc, na_position="last")
            .reset_index(drop=True)
            .head(max_rows)
        )
        st.dataframe(display_df, width="stretch")

    # ---------------------------------------------------------------------------
    # PAGE 3: Statistics
    # ---------------------------------------------------------------------------
    elif page == "Statistics":
        st.title("📈 Airport Statistics")
        st.markdown(
            "Aggregated statistics and insights about the current filtered airport set. "
            "Adjust the sidebar to compare different subsets."
        )

        if filtered_df.empty:
            st.warning("No airports match the current filters.")
            st.stop()

        # Pie chart: type proportions
        st.subheader("🥧 Airport Type Breakdown")
        type_counts = filtered_df["type"].value_counts()
        palette = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        _, texts, autotexts = ax3.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            colors=palette,
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax3.set_title("Distribution of Airport Types", fontsize=13, fontweight="bold", pad=14)
        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown("---")

        # Continent bar chart - second call to count_by_group  #[FUNCCALL2]
        st.subheader("🌐 Airports by Continent")
        cont_counts = count_by_group(filtered_df, "continent")
        cont_counts.index = cont_counts.index.map(lambda c: CONTINENT_NAMES.get(c, c))

        fig4, ax4 = plt.subplots(figsize=(9, 4))
        cont_colors = plt.cm.Pastel1(np.linspace(0, 1, len(cont_counts)))
        bars4 = ax4.barh(cont_counts.index, cont_counts.values,
                         color=cont_colors, edgecolor="gray", linewidth=0.5)
        ax4.set_title("Airport Count by Continent", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Number of Airports")
        ax4.spines[["top", "right"]].set_visible(False)
        ax4.grid(axis="x", linestyle="--", alpha=0.4)
        for bar, val in zip(bars4, cont_counts.values):
            ax4.text(
                bar.get_width() + max(cont_counts.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9,
            )
        plt.tight_layout()
        st.pyplot(fig4)

        st.markdown("---")

        # Elevation extremes  #[MAXMIN] #[FUNCRETURN2]
        st.subheader("⛰️ Elevation Extremes")
        highest, lowest = get_elevation_extremes(filtered_df)

        if highest is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.success(
                    f"**Highest Airport**\n\n"
                    f"**{highest['name']}**\n\n"
                    f"{highest['municipality']}, {highest['iso_country']}\n\n"
                    f"Elevation: **{int(highest['elevation_ft']):,} ft**\n\n"
                    f"Type: {highest['type']}"
                )
            with col_b:
                st.info(
                    f"**Lowest Airport**\n\n"
                    f"**{lowest['name']}**\n\n"
                    f"{lowest['municipality']}, {lowest['iso_country']}\n\n"
                    f"Elevation: **{int(lowest['elevation_ft']):,} ft**\n\n"
                    f"Type: {lowest['type']}"
                )

        # Elevation histogram
        elev_data = filtered_df["elevation_ft"].dropna()
        if len(elev_data) > 10:
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            ax5.hist(elev_data, bins=60, color="#2a7abd", edgecolor="white",
                     linewidth=0.4, alpha=0.85)
            ax5.set_title("Distribution of Airport Elevations", fontsize=13, fontweight="bold")
            ax5.set_xlabel("Elevation (ft)")
            ax5.set_ylabel("Number of Airports")
            ax5.axvline(elev_data.mean(),   color="red",    linestyle="--",
                        linewidth=1.2, label=f"Mean: {elev_data.mean():.0f} ft")
            ax5.axvline(elev_data.median(), color="orange", linestyle="--",
                        linewidth=1.2, label=f"Median: {elev_data.median():.0f} ft")
            ax5.legend(fontsize=9)
            ax5.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig5)

        st.markdown("---")

        # Dictionary methods  #[DICTMETHOD]
        st.subheader("📖 Airport Type Reference")
        type_descriptions = {
            "large_airport":  "Scheduled international or major domestic service.",
            "medium_airport": "Regional airports with scheduled service to major hubs.",
            "small_airport":  "Local airports, often private or general aviation.",
            "heliport":       "Designated landing areas for helicopters only.",
            "seaplane_base":  "Water-based takeoff and landing facilities.",
            "closed":         "Airports that are no longer in service.",
            "balloonport":    "Designated areas for hot-air balloon operations.",
        }

        # .items() - one dictionary method  #[DICTMETHOD]
        for type_name, description in type_descriptions.items():
            count = int((filtered_df["type"] == type_name).sum())
            if count > 0:
                st.markdown(f"**{type_name}** ({count:,} in current filter): {description}")

        # .keys() - second dictionary method  #[DICTMETHOD]
        present_keys = [k for k in type_descriptions.keys() if k in filtered_df["type"].values]
        st.caption(f"Types present in current filter: {', '.join(present_keys) if present_keys else 'none'}")

        st.markdown("---")

        # Country summary loop  #[ITERLOOP]
        st.subheader("🔢 Country Summary Table")
        st.markdown("Airports per country broken down by type (top 15 by total).")

        summary_rows = []
        # #[ITERLOOP] - Loop that iterates through grouped DataFrame items
        for country_code, group in filtered_df.groupby("iso_country"):
            avg_elev = round(group["elevation_ft"].dropna().mean(), 0)
            summary_rows.append({
                "Country":       country_code,
                "Continent":     CONTINENT_NAMES.get(group["continent"].iloc[0], ""),
                "Total":         len(group),
                "Large":         int((group["type"] == "large_airport").sum()),
                "Medium":        int((group["type"] == "medium_airport").sum()),
                "Small":         int((group["type"] == "small_airport").sum()),
                "Heliport":      int((group["type"] == "heliport").sum()),
                "Avg Elev (ft)": avg_elev if not np.isnan(avg_elev) else None,
            })

        summary_df = pd.DataFrame(summary_rows)
        # #[SORT] - Sort country summary by total descending
        summary_df = summary_df.sort_values("Total", ascending=False).reset_index(drop=True)
        st.dataframe(summary_df.head(15), width="stretch")



if __name__ == "__main__":
    main()
