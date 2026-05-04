import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RQ1 — Federal R&D Flow & Foreign Collaboration", layout="wide")

API_BASE      = "https://api.usaspending.gov/api/v2"
OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_MAIL = "em1798@msstate.edu"

AGENCIES = {
    "Department of Defense":                   "DoD",
    "Department of Energy":                    "DOE",
    "Department of Health and Human Services": "HHS",
    "Department of Homeland Security":         "DHS",
    "National Science Foundation":             "NSF",
}

RECIPIENT_TYPES = [
    "public_institution_of_higher_education",
    "private_institution_of_higher_education",
    "minority_serving_institution_of_higher_education",
]

FISCAL_YEARS = list(range(2010, 2025))  # FY2010–FY2024, 15 complete years

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC","PR","VI","GU","AS","MP",
]

GEOCODE_CACHE_FILE = "geocode_cache.json"

# US territories — treated as domestic in foreign collaboration analysis
US_TERRITORIES = {"US", "PR", "GU", "VI", "AS", "MP"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def format_dollars(val):
    if pd.isna(val) or val == 0:
        return "—"
    if abs(val) >= 1e9:
        return f"${val/1e9:,.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:,.2f}M"
    if abs(val) >= 1e3:
        return f"${val/1e3:,.1f}K"
    return f"${val:,.0f}"


def fy_to_dates(fy):
    return {"start_date": f"{fy-1}-10-01", "end_date": f"{fy}-09-30"}


# ── Geocoding (reused from your existing dashboard) ──────────────────────────
def load_geocode_cache():
    if os.path.exists(GEOCODE_CACHE_FILE):
        with open(GEOCODE_CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_geocode_cache(cache):
    with open(GEOCODE_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_universities(names: tuple) -> dict:
    cache   = load_geocode_cache()
    missing = [n for n in names if n not in cache]
    if missing:
        geolocator = Nominatim(user_agent="rq1_dashboard_v1", timeout=10)
        geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)
        prog       = st.progress(0, text="Geocoding universities...")
        for i, name in enumerate(missing):
            prog.progress(i / len(missing), text=f"Geocoding: {name[:50]}...")
            try:
                loc = geocode(name, country_codes="us")
                cache[name] = {"lat": loc.latitude, "lon": loc.longitude} if loc else None
            except Exception:
                cache[name] = None
        prog.empty()
        save_geocode_cache(cache)
    return {n: cache.get(n) for n in names}


# ── TAB 1: University obligations (your existing pattern) ────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recipients_for_fy(agency_name, fy, state_code=None, limit=100):
    """spending_by_category/recipient — aggregated totals per university."""
    dates   = fy_to_dates(fy)
    filters = {
        "time_period":          [dates],
        "agencies":             [{"type": "funding", "tier": "toptier", "name": agency_name}],
        "recipient_type_names": RECIPIENT_TYPES,
    }
    if state_code:
        filters["recipient_locations"] = [{"country": "USA", "state": state_code}]

    all_results, page = [], 1
    while True:
        payload = {"filters": filters, "category": "recipient", "limit": limit, "page": page}
        try:
            resp = requests.post(
                f"{API_BASE}/search/spending_by_category/recipient",
                json=payload, timeout=60
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as e:
            st.error(f"USAspending error (FY{fy}, {agency_name}, p{page}): {e}")
            break
        if not results:
            break
        all_results.extend(results)
        if len(results) < limit:
            break
        page += 1
        time.sleep(0.3)
    return all_results


def build_university_dataframe(agency_names, fiscal_years, state_code=None):
    records = []
    total_steps = len(agency_names) * len(fiscal_years)
    prog = st.progress(0, text="Fetching university obligations...")
    step = 0
    for agency_name in agency_names:
        abbr = AGENCIES[agency_name]
        for fy in fiscal_years:
            prog.progress(step / total_steps, text=f"Fetching {abbr} FY{fy}...")
            results = fetch_recipients_for_fy(agency_name, fy, state_code)
            for r in results:
                records.append({
                    "Recipient":       r.get("name", "Unknown"),
                    "Agency":          abbr,
                    "FY":              f"FY{fy}",
                    "Obligations ($)": r.get("amount", 0),
                })
            step += 1
    prog.empty()

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    df_raw = pd.DataFrame(records)
    pivot  = df_raw.pivot_table(
        index="Recipient", columns="FY",
        values="Obligations ($)", aggfunc="sum", fill_value=0,
    )
    fy_cols = [f"FY{y}" for y in fiscal_years if f"FY{y}" in pivot.columns]
    pivot   = pivot[fy_cols]
    pivot["Total"]      = pivot[fy_cols].sum(axis=1)
    recent_3            = fy_cols[-3:] if len(fy_cols) >= 3 else fy_cols
    pivot["3-Yr Avg"]   = pivot[recent_3].replace(0, pd.NA).mean(axis=1, skipna=True).fillna(0)
    pivot["Annual Avg"] = pivot[fy_cols].replace(0, pd.NA).mean(axis=1, skipna=True).fillna(0)
    pivot = pivot.sort_values("Total", ascending=False)

    agency_pivot = df_raw.pivot_table(
        index="Recipient", columns="Agency",
        values="Obligations ($)", aggfunc="sum", fill_value=0,
    )
    return pivot, agency_pivot


# ── TAB 2: Award ID → OpenAlex foreign collaboration ─────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_award_ids_for_fy(agency_name, fy, state_code=None, limit=100, max_records=5000):
    """
    spending_by_award — returns individual award rows WITH Award ID.
    This is the join key to OpenAlex awards[].funder_award_id.
    Capped at max_records to avoid connection timeouts.
    State filter applied via recipient_locations when provided.
    """
    dates   = fy_to_dates(fy)
    filters = {
        "time_period":          [dates],
        "agencies":             [{"type": "funding", "tier": "toptier", "name": agency_name}],
        "recipient_type_names": RECIPIENT_TYPES,
        "award_type_codes":     ["02", "03", "04", "05"],  # grants only
    }
    if state_code:
        filters["recipient_locations"] = [{"country": "USA", "state": state_code}]
    fields = ["Award ID", "Recipient Name", "Award Amount", "Award Type"]

    all_results, page = [], 1
    while len(all_results) < max_records:
        payload = {
            "filters": filters, "fields": fields,
            "limit": limit, "page": page,
            "sort": "Award Amount", "order": "desc",
            "subawards": False,
        }
        success = False
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{API_BASE}/search/spending_by_award/",
                    json=payload, timeout=60
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                success = True
                break
            except Exception as e:
                time.sleep(2 ** attempt)
        if not success:
            break
        if not results:
            break
        all_results.extend(results)
        if len(results) < limit:
            break
        page += 1
        time.sleep(0.5)

    return all_results[:max_records]


def build_award_dataframe(agency_names, fiscal_years, state_code=None):
    """Collect individual awards with Award IDs across agencies and FYs."""
    records = []
    total   = len(agency_names) * len(fiscal_years)
    prog    = st.progress(0, text="Fetching individual awards...")
    step    = 0

    for agency_name in agency_names:
        abbr = AGENCIES[agency_name]
        for fy in fiscal_years:
            prog.progress(step / total, text=f"Awards: {abbr} FY{fy}...")
            results = fetch_award_ids_for_fy(agency_name, fy, state_code=state_code)
            for r in results:
                award_id = (r.get("Award ID") or "").strip().upper()
                if award_id:
                    records.append({
                        "award_id":       award_id,
                        "recipient_name": r.get("Recipient Name", "Unknown"),
                        "agency":         abbr,
                        "fiscal_year":    fy,
                        "award_amount":   r.get("Award Amount", 0),
                    })
            step += 1

    prog.empty()
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["award_id"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openalex_for_award(award_id: str) -> list:
    """
    Query OpenAlex for publications that acknowledged a specific award.
    Uses: filter=awards.funder_award_id:{award_id}
    Returns list of works with authorship country codes.

    NOTE: OpenAlex field is `awards` (NOT `grants`).
          Join key is `awards[].funder_award_id` (NOT `award_id`).
    """
    try:
        resp = requests.get(
            f"{OPENALEX_BASE}/works",
            params={
                "filter":   f"awards.funder_award_id:{award_id}",
                "select":   "id,title,publication_year,authorships,awards,fwci",
                "per_page": 50,
                "mailto":   OPENALEX_MAIL,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception:
        return []


def build_foreign_collab_dataframe(award_df, sample_size=200):
    """
    For each award ID, query OpenAlex and extract foreign collaboration data.
    Joins back to USAspending recipient and amount.
    Samples top awards by amount to stay within API rate limits.
    """
    if award_df.empty:
        return pd.DataFrame()

    # Sample top awards by amount — largest grants most likely to have publications
    sample = award_df.nlargest(sample_size, "award_amount")

    rows   = []
    prog   = st.progress(0, text="Querying OpenAlex for publications...")
    total  = len(sample)

    for i, (_, award) in enumerate(sample.iterrows()):
        prog.progress(i / total, text=f"OpenAlex: {award['award_id']}...")
        works = fetch_openalex_for_award(award["award_id"])

        pub_count         = len(works)
        foreign_countries = set()
        total_authorships = 0
        foreign_count     = 0
        fwci_values       = []

        for work in works:
            # Collect FWCI per paper
            fwci = work.get("fwci")
            if fwci is not None:
                fwci_values.append(fwci)

            for authorship in work.get("authorships", []):
                for inst in authorship.get("institutions", []):
                    cc = inst.get("country_code")
                    if cc:
                        total_authorships += 1
                        if cc not in US_TERRITORIES:
                            foreign_count += 1
                            foreign_countries.add(cc)

        foreign_pct = (
            round(foreign_count / total_authorships * 100, 1)
            if total_authorships > 0 else 0.0
        )

        # Average FWCI across all publications under this award
        # Default to 1.0 (average impact) if no FWCI data available
        avg_fwci = round(sum(fwci_values) / len(fwci_values), 3) if fwci_values else 1.0

        # IP Exposure ($) — dollar-weighted foreign exposure (no FWCI)
        ip_exposure = round(award["award_amount"] * foreign_pct / 100, 2)

        # FKEI — Foreign Knowledge Exposure Index
        # = Award Amount × Foreign Collab % × Avg FWCI
        # Dollar primary (policy variable), FWCI adjusts for knowledge impact
        fkei = round(award["award_amount"] * foreign_pct / 100 * avg_fwci, 2)

        rows.append({
            "Award ID":            award["award_id"],
            "University":          award["recipient_name"],
            "Agency":              award["agency"],
            "Fiscal Year":         award["fiscal_year"],
            "Award Amount ($)":    award["award_amount"],
            "Publications":        pub_count,
            "Avg FWCI":            avg_fwci,
            "Foreign Collab %":    foreign_pct,
            "Foreign Countries":   ", ".join(sorted(foreign_countries)),
            "N Foreign Countries": len(foreign_countries),
            "IP Exposure ($)":     ip_exposure,
            "FKEI":                fkei,
        })
        time.sleep(0.1)

    prog.empty()
    df = pd.DataFrame(rows)

    # Aggregate to university level
    if df.empty:
        return df

    return df


def aggregate_to_university(df):
    """Roll up award-level data to university level for charts."""
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("University").agg(
        Total_Awards       =("Award ID",         "count"),
        Total_Amount       =("Award Amount ($)",  "sum"),
        Total_Publications =("Publications",      "sum"),
        Avg_Foreign_Pct    =("Foreign Collab %",  "mean"),
        Avg_FWCI           =("Avg FWCI",          "mean"),
        Total_IP_Exposure  =("IP Exposure ($)",   "sum"),
        Total_FKEI         =("FKEI",              "sum"),
        All_Countries      =("Foreign Countries", lambda x: ", ".join(
            sorted(set(c for v in x for c in v.split(", ") if c))
        )),
    ).reset_index()
    agg["Avg_Foreign_Pct"] = agg["Avg_Foreign_Pct"].round(1)
    agg["Avg_FWCI"]        = agg["Avg_FWCI"].round(3)
    agg["Total_IP_Exposure"] = agg["Total_IP_Exposure"].round(2)
    agg["Total_FKEI"]      = agg["Total_FKEI"].round(2)
    return agg.sort_values("Total_Amount", ascending=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Filters")

agency_mode = st.sidebar.radio("Agency Selection", ["Single Agency", "All Agencies"])
if agency_mode == "Single Agency":
    selected_agency = st.sidebar.selectbox("Agency", list(AGENCIES.keys()), index=4)
    agency_list     = [selected_agency]
    display_label   = AGENCIES[selected_agency]
else:
    agency_list   = list(AGENCIES.keys())
    display_label = "All Agencies"

fy_range = st.sidebar.multiselect(
    "Fiscal Years", FISCAL_YEARS, default=list(range(2018, 2025))
)

state_filter = st.sidebar.selectbox(
    "University State", ["All States"] + US_STATES
)
state_code = None if state_filter == "All States" else state_filter

top_n = st.sidebar.slider("Top N universities", 10, 100, 25, 5)

st.sidebar.markdown("---")
openalex_sample = st.sidebar.slider(
    "OpenAlex sample size (Tab 2)", 50, 500, 200, 50,
    help="Number of top awards by dollar amount to query in OpenAlex. Higher = more complete but slower."
)

fetch_choice = st.sidebar.selectbox(
    "Select data to fetch",
    ["🏛️ University Obligations", "🌐 Foreign Collaboration (FKEI)"],
)

fetch_btn  = st.sidebar.button("Fetch Data", type="primary", use_container_width=True)
fetch_tab1 = fetch_btn and fetch_choice == "🏛️ University Obligations"
fetch_tab2 = fetch_btn and fetch_choice == "🌐 Foreign Collaboration (FKEI)"

st.sidebar.markdown("---")
st.sidebar.caption("Sources: USAspending.gov · OpenAlex.org")
st.sidebar.caption("Join: USAspending `Award ID` == OpenAlex `awards[].funder_award_id`")
st.sidebar.caption("Index: FKEI = Award $ × Foreign Collab % × Avg FWCI")


# ── Title ─────────────────────────────────────────────────────────────────────
st.title(f"RQ1 — Federal R&D Flow & Foreign Collaboration — {display_label}")
st.caption("FY2010–FY2024 · Higher Education Recipients · USAspending + OpenAlex")

if not fy_range:
    st.warning("Select at least one fiscal year.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🏛️ University Obligations (USAspending)",
    "🌐 Foreign Collaboration (OpenAlex Join)",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — University Obligations (your existing dashboard pattern)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if fetch_tab1:
        with st.spinner("Querying USAspending API..."):
            df_pivot, df_agency = build_university_dataframe(
                agency_list, sorted(fy_range), state_code
            )
        if df_pivot.empty:
            st.warning("No data returned.")
        else:
            st.session_state["df_pivot"]  = df_pivot
            st.session_state["df_agency"] = df_agency
            st.session_state["t1_label"]  = display_label

    if "df_pivot" in st.session_state:
        df_pivot  = st.session_state["df_pivot"]
        df_agency = st.session_state["df_agency"]
        label     = st.session_state.get("t1_label", display_label)

        fy_cols = [c for c in df_pivot.columns if c.startswith("FY")]

        # KPI row
        total_obl  = df_pivot["Total"].sum()
        num_univs  = len(df_pivot)
        top_univ   = df_pivot.index[0] if num_univs > 0 else "N/A"
        top_amount = df_pivot["Total"].iloc[0] if num_univs > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Obligations",  format_dollars(total_obl))
        c2.metric("Universities",        f"{num_univs:,}")
        c3.metric("Top Recipient",       top_univ)
        c4.metric("Top Recipient Amount", format_dollars(top_amount))

        st.divider()

        # Map
        st.subheader("University Locations — Federal Obligations")
        univ_names  = df_pivot.head(top_n).index.tolist()
        geo_results = geocode_universities(tuple(univ_names))

        map_records = []
        for name in univ_names:
            coords = geo_results.get(name)
            if coords:
                row = df_pivot.loc[name]
                map_records.append({
                    "University": name,
                    "lat":        coords["lat"],
                    "lon":        coords["lon"],
                    "Total":      row["Total"],
                    "3-Yr Avg":   row["3-Yr Avg"],
                })

        if map_records:
            map_df  = pd.DataFrame(map_records)
            fig_map = px.scatter_geo(
                map_df, lat="lat", lon="lon",
                size="Total", color="3-Yr Avg",
                hover_name="University",
                hover_data={"Total": ":$,.0f", "3-Yr Avg": ":$,.0f", "lat": False, "lon": False},
                color_continuous_scale="Blues", size_max=40, scope="usa",
            )
            fig_map.update_layout(
                height=500,
                geo=dict(showland=True, landcolor="rgb(243,243,243)",
                         showlakes=True, lakecolor="rgb(204,224,245)",
                         showsubunits=True, subunitcolor="rgb(200,200,200)"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        st.divider()

        # Bar chart — total obligations
        st.subheader(f"Top {min(top_n, num_univs)} Universities — Total Obligations")
        chart_df = df_pivot.head(top_n).reset_index()
        fig_bar  = px.bar(
            chart_df, x="Total", y="Recipient", orientation="h",
            color="3-Yr Avg", color_continuous_scale="Blues",
            labels={"Total": "Total Obligations ($)", "Recipient": ""},
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=max(400, top_n * 28))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Year-over-year trend
        st.subheader("Year-over-Year Trend — Top 10 Universities")
        trend_df = df_pivot.head(10)[fy_cols].reset_index().melt(
            id_vars="Recipient", var_name="Fiscal Year", value_name="Obligations ($)"
        )
        fig_line = px.line(
            trend_df, x="Fiscal Year", y="Obligations ($)",
            color="Recipient", markers=True,
        )
        fig_line.update_layout(
            height=500,
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Heatmap
        st.subheader("Obligation Heatmap")
        heatmap_data = df_pivot.head(top_n)[fy_cols]
        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            colorscale="Blues",
            hovertemplate="<b>%{y}</b><br>%{x}: $%{z:,.0f}<extra></extra>",
        ))
        fig_heat.update_layout(
            height=max(400, min(top_n, num_univs) * 28),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Agency breakdown
        if len(agency_list) > 1 and not df_agency.empty:
            st.subheader("Obligation Breakdown by Agency")
            agency_display = df_agency.loc[
                df_agency.index.isin(df_pivot.head(top_n).index)
            ].copy()
            agency_display["Total"] = agency_display.sum(axis=1)
            agency_display = agency_display.sort_values("Total", ascending=False)
            fmt_agency = agency_display.copy()
            for col in fmt_agency.columns:
                fmt_agency[col] = fmt_agency[col].apply(format_dollars)
            st.dataframe(fmt_agency, use_container_width=True, height=400)

        # Full table
        st.subheader("Full Data Table")
        display_df = df_pivot.head(top_n).copy()
        fmt_df = display_df.copy()
        for col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].apply(format_dollars)
        st.dataframe(fmt_df, use_container_width=True, height=500)

        # Download
        csv = df_pivot.reset_index().to_csv(index=False)
        st.download_button(
            "Download CSV", csv,
            file_name=f"university_obligations_{label}_FY{min(fy_range)}_FY{max(fy_range)}.csv",
            mime="text/csv",
        )
    else:
        st.info("Click **Fetch University Obligations** in the sidebar to load data.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Foreign Collaboration via OpenAlex Award ID Join
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    **How this works:**
    1. USAspending `spending_by_award` → individual awards with **Award ID**
    2. OpenAlex `awards[].funder_award_id` → publications per award
    3. Join on Award ID (exact match) → foreign co-author country codes
    4. Aggregate to university level → IP exposure profile
    """)
    st.caption("OpenAlex field: `awards[].funder_award_id` · USAspending field: `Award ID`")
    st.divider()

    if fetch_tab2:
        # Step 1 — get individual awards with Award IDs
        with st.spinner("Fetching individual awards from USAspending..."):
            award_df = build_award_dataframe(agency_list, sorted(fy_range), state_code=state_code)

        if award_df.empty:
            st.warning("No awards returned. Try adjusting filters.")
        else:
            st.success(f"Found {len(award_df):,} unique awards. Querying OpenAlex...")

            # Step 2 — query OpenAlex for each award ID
            collab_df = build_foreign_collab_dataframe(award_df, sample_size=openalex_sample)

            if collab_df.empty:
                st.warning("No OpenAlex matches found. Award IDs may not be indexed yet.")
            else:
                st.session_state["collab_df"]  = collab_df
                st.session_state["award_df"]   = award_df
                st.session_state["t2_label"]   = display_label

    if "collab_df" in st.session_state:
        collab_df = st.session_state["collab_df"]
        award_df  = st.session_state["award_df"]

        # Matched awards stats
        matched   = collab_df[collab_df["Publications"] > 0]
        unmatched = collab_df[collab_df["Publications"] == 0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Awards Sampled",        f"{len(collab_df):,}")
        c2.metric("Awards Matched (OpenAlex)", f"{len(matched):,}")
        c3.metric("Total Publications",    f"{collab_df['Publications'].sum():,}")
        c4.metric("Match Rate",            f"{len(matched)/len(collab_df)*100:.1f}%")

        st.divider()

        # Aggregate to university level
        univ_df = aggregate_to_university(matched)

        if not univ_df.empty:

            # Scatter — federal obligation vs foreign collab %
            st.subheader("Federal Obligation vs. Foreign Collaboration Share")
            st.caption("Each bubble = one university. Size = total award amount. Color = foreign collab %.")
            fig_sc = px.scatter(
                univ_df,
                x="Total_Amount",
                y="Avg_Foreign_Pct",
                size="Total_Amount",
                color="Avg_Foreign_Pct",
                hover_name="University",
                hover_data={
                    "Total_Amount":       ":$,.0f",
                    "Avg_Foreign_Pct":    ":.1f",
                    "Avg_FWCI":           ":.3f",
                    "Total_FKEI":         ":$,.0f",
                    "Total_Publications": True,
                    "All_Countries":      True,
                },
                color_continuous_scale="RdYlGn_r",
                size_max=50,
                labels={
                    "Total_Amount":    "Federal Award Amount ($)",
                    "Avg_Foreign_Pct": "Foreign Co-author Share (%)",
                },
            )
            fig_sc.update_layout(
                height=520,
                xaxis=dict(showgrid=True, gridcolor="#eee"),
                yaxis=dict(showgrid=True, gridcolor="#eee"),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            st.divider()

            col_l, col_r = st.columns(2)

            with col_l:
                # Bar — universities ranked by foreign collab %
                st.subheader("Universities Ranked by Foreign Collaboration %")
                rank_df  = univ_df.sort_values("Avg_Foreign_Pct", ascending=True).head(top_n)
                fig_rank = px.bar(
                    rank_df, x="Avg_Foreign_Pct", y="University",
                    orientation="h",
                    color="Avg_Foreign_Pct",
                    color_continuous_scale="RdYlGn_r",
                    text=rank_df["Avg_Foreign_Pct"].apply(lambda x: f"{x:.1f}%"),
                    labels={"Avg_Foreign_Pct": "Foreign Co-author %", "University": ""},
                )
                fig_rank.update_traces(textposition="outside")
                fig_rank.update_layout(
                    height=max(400, len(rank_df) * 30),
                    coloraxis_showscale=False,
                    plot_bgcolor="white",
                    margin=dict(l=0, r=60, t=20, b=10),
                )
                st.plotly_chart(fig_rank, use_container_width=True)

            with col_r:
                # Bar — universities ranked by total publications
                st.subheader("Universities Ranked by Publication Count")
                pub_df  = univ_df.sort_values("Total_Publications", ascending=True).head(top_n)
                fig_pub = px.bar(
                    pub_df, x="Total_Publications", y="University",
                    orientation="h",
                    color="Total_Publications",
                    color_continuous_scale="Blues",
                    text="Total_Publications",
                    labels={"Total_Publications": "Publications", "University": ""},
                )
                fig_pub.update_traces(textposition="outside")
                fig_pub.update_layout(
                    height=max(400, len(pub_df) * 30),
                    coloraxis_showscale=False,
                    plot_bgcolor="white",
                    margin=dict(l=0, r=60, t=20, b=10),
                )
                st.plotly_chart(fig_pub, use_container_width=True)

            st.divider()

            # Country frequency chart
            st.subheader("Top Foreign Collaborating Countries")
            country_counts = {}
            for countries_str in matched["Foreign Countries"]:
                for c in countries_str.split(", "):
                    if c.strip():
                        country_counts[c.strip()] = country_counts.get(c.strip(), 0) + 1

            if country_counts:
                country_df  = pd.DataFrame(
                    sorted(country_counts.items(), key=lambda x: -x[1]),
                    columns=["Country", "Count"]
                ).head(20)
                fig_country = px.bar(
                    country_df, x="Count", y="Country",
                    orientation="h",
                    color="Count",
                    color_continuous_scale="Oranges",
                    labels={"Count": "Number of Awards with Collaboration", "Country": ""},
                )
                fig_country.update_layout(
                    height=500,
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig_country, use_container_width=True)

            st.divider()

            # FKEI chart — primary novel contribution
            st.subheader("Universities Ranked by FKEI — Foreign Knowledge Exposure Index")
            st.caption(
                "**FKEI = Award Amount ($) × Foreign Collab % × Avg FWCI** — "
                "Dollar is the primary policy variable. "
                "FWCI adjusts for research impact — high-impact knowledge shared "
                "with foreign collaborators scores higher."
            )

            fkei_df = univ_df.copy()
            fkei_df = fkei_df.sort_values("Total_FKEI", ascending=True).head(top_n)

            fig_fkei = px.bar(
                fkei_df,
                x="Total_FKEI", y="University",
                orientation="h",
                color="Total_FKEI",
                color_continuous_scale="Reds",
                text=fkei_df["Total_FKEI"].apply(format_dollars),
                labels={"Total_FKEI": "FKEI Score ($)", "University": ""},
                hover_data={
                    "Total_Amount":    ":$,.0f",
                    "Avg_Foreign_Pct": ":.1f",
                    "Avg_FWCI":        ":.3f",
                    "Total_FKEI":      ":$,.0f",
                },
            )
            fig_fkei.update_traces(textposition="outside")
            fig_fkei.update_layout(
                height=max(400, len(fkei_df) * 30),
                coloraxis_showscale=False,
                plot_bgcolor="white",
                margin=dict(l=0, r=80, t=20, b=10),
                xaxis_title="FKEI Score ($)",
                yaxis_title="",
            )
            st.plotly_chart(fig_fkei, use_container_width=True)

            # FKEI summary KPIs
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Total FKEI Score",
                format_dollars(fkei_df["Total_FKEI"].sum()),
                help="Sum of FKEI across top universities — total dollar-impact exposure"
            )
            c2.metric(
                "Avg FWCI (Top Universities)",
                f"{univ_df['Avg_FWCI'].mean():.2f}",
                help="Average field-weighted citation impact. >1.0 = above global average"
            )
            c3.metric(
                "FKEI vs IP Exposure Uplift",
                f"{(univ_df['Total_FKEI'].sum() / univ_df['Total_IP_Exposure'].sum()):.2f}x",
                help="How much FWCI amplifies raw IP Exposure — impact adjustment factor"
            )

            # Award-level detail table
            st.subheader("Award-Level Detail Table")
            st.caption("One row per award. Award ID is the join key between USAspending and OpenAlex.")
            display_cols = [
                "Award ID", "University", "Agency", "Fiscal Year",
                "Award Amount ($)", "Publications", "Avg FWCI",
                "Foreign Collab %", "IP Exposure ($)", "FKEI", "Foreign Countries"
            ]
            detail_df = matched[display_cols].sort_values("Foreign Collab %", ascending=False)
            st.dataframe(detail_df, use_container_width=True, height=500)

            # Unmatched awards
            if not unmatched.empty:
                with st.expander(f"Awards with no OpenAlex match ({len(unmatched):,})"):
                    st.caption("These awards have no publications indexed in OpenAlex yet.")
                    st.dataframe(
                        unmatched[["Award ID", "University", "Agency", "Award Amount ($)"]],
                        use_container_width=True, height=300
                    )

            # Download
            csv = collab_df.to_csv(index=False)
            st.download_button(
                "Download Foreign Collaboration CSV",
                csv,
                file_name=f"foreign_collab_{display_label}_FY{min(fy_range)}_FY{max(fy_range)}.csv",
                mime="text/csv",
            )

    else:
        st.info("Click **Fetch Foreign Collaboration** in the sidebar to load data.")
        st.markdown("""
        **What this tab will show:**
        - Scatter plot: federal obligation vs foreign collaboration %
        - Universities ranked by foreign co-author share
        - Top foreign collaborating countries
        - Award-level detail table with OpenAlex match status
        - Download CSV of full results
        """)

st.caption("Data: USAspending.gov · OpenAlex.org | ECE Dept, Mississippi State University")
