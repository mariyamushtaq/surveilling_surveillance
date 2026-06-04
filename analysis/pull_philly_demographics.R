#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(sf)
  library(tidycensus)
  library(here)
})

# Pull ACS 5-year block-group demographics for Philadelphia County,
# compute Shannon entropy + regression covariates, and project to detection CRS.

acs_year   <- 2024
acs_survey <- "acs5"

key <- Sys.getenv("CENSUS_API_KEY", unset = "")
if (key == "") {
  stop(
    paste(
      "CENSUS_API_KEY is not set.",
      "Set it before running:",
      "export CENSUS_API_KEY='your_key_here'"
    ),
    call. = FALSE
  )
}

tidycensus::census_api_key(key, install = FALSE, overwrite = TRUE)

# Load the exact city polygon used for camera analysis sampling
# Place philadelphia_city_wgs84.geojson in data/input_metadata/ (same dir as philly_meta.csv)
city_polygon_path <- here::here("data", "input_metadata", "philadelphia_city_wgs84.geojson")
if (!file.exists(city_polygon_path)) {
  stop("Missing required file: data/input_metadata/philadelphia_city_wgs84.geojson", call. = FALSE)
}
city_polygon <- st_read(city_polygon_path, quiet = TRUE)
message("City polygon loaded: ", nrow(city_polygon), " feature(s), ",
        nchar(st_as_text(st_geometry(city_polygon)[[1]])) %/% 1000, "k-char WKT")

message("Reading detection/sample metadata to infer target CRS...")

meta_path <- here::here("data", "input_metadata", "philly_meta.csv")
if (!file.exists(meta_path)) {
  stop("Missing required file: data/input_metadata/philly_meta.csv", call. = FALSE)
}

meta <- readr::read_csv(meta_path, show_col_types = FALSE)

if (!all(c("lon_anchor", "lat_anchor") %in% names(meta))) {
  if (all(c("lon", "lat") %in% names(meta))) {
    meta <- meta %>% mutate(lon_anchor = lon, lat_anchor = lat)
  } else {
    stop("Metadata must include lon_anchor/lat_anchor or lon/lat columns.", call. = FALSE)
  }
}

detection_points <- meta %>%
  st_as_sf(coords = c("lon_anchor", "lat_anchor"), crs = 4326, remove = FALSE)

target_crs <- st_crs(detection_points)
message("Target CRS detected: ", target_crs$input)

message("Pulling ACS block-group data for Philadelphia County...")

# Race/ethnicity subgroups for Shannon entropy (all from B03002)
# FIX: B17001 (poverty) removed — unavailable at block-group level in ACS5 2024.
# Use poverty_income_ratio (C17002) instead; see note below.
vars <- c(
  total_pop                 = "B03002_001",
  nh_white                  = "B03002_003",
  nh_black                  = "B03002_004",
  nh_native                 = "B03002_005",
  nh_asian                  = "B03002_006",
  nh_pacific                = "B03002_007",
  nh_other                  = "B03002_008",
  nh_multiracial            = "B03002_009",
  hispanic                  = "B03002_012",
  median_household_income   = "B19013_001",
  # C17002: ratio of income to poverty (available at BG level; replaces B17001)
  # _002 = < 0.50, _003 = 0.50-0.99 → sum = population below poverty line
  poverty_ratio_total       = "C17002_001",
  poverty_ratio_below_half  = "C17002_002",
  poverty_ratio_half_to_1   = "C17002_003",
  labor_force               = "B23025_003",
  unemployed                = "B23025_005",
  housing_units             = "B25001_001",
  occupied_housing_units    = "B25002_002",
  median_gross_rent         = "B25064_001",
  median_home_value         = "B25077_001",
  total_education_25_plus   = "B15003_001",
  bachelors                 = "B15003_022",
  masters                   = "B15003_023",
  professional              = "B15003_024",
  doctorate                 = "B15003_025"
)

acs_long <- tidycensus::get_acs(
  geography = "block group",
  state     = "PA",
  county    = "Philadelphia County",
  survey    = acs_survey,
  year      = acs_year,
  variables = vars,
  geometry  = TRUE,
  keep_geo_vars = TRUE
)

# Separate geometry before any reshape — avoids sticky-geometry contamination
# in later pmap calls.
acs_geo <- acs_long %>%
  st_as_sf() %>%
  select(any_of(c("GEOID", "NAME", "NAMELSAD")), geometry) %>%
  distinct(GEOID, .keep_all = TRUE)

# Spatially filter to the exact city polygon used for camera analysis.
# st_filter with .predicate = st_intersects keeps any block group whose
# geometry touches or overlaps the polygon — same footprint as the GSV pull.
# The city polygon is WGS84 (4326); acs_geo from tidycensus is NAD83 (4269),
# effectively identical at this scale but we reproject to be explicit.
city_polygon_4269 <- st_transform(city_polygon, st_crs(acs_geo))
acs_geo_filtered  <- st_filter(acs_geo, city_polygon_4269, .predicate = st_intersects)
message("Block groups after city polygon filter: ", nrow(acs_geo_filtered),
        " (from ", n_distinct(acs_long$GEOID), " county-wide)")

valid_geoids <- acs_geo_filtered$GEOID
acs_geo      <- acs_geo_filtered   # use filtered set from here on

# Pivot to wide format on the attribute-only (geometry-dropped) data
acs_wide <- acs_long %>%
  st_drop_geometry() %>%                               # drop before pivot
  select(GEOID, variable, estimate, moe) %>%
  pivot_wider(
    id_cols     = GEOID,
    names_from  = variable,
    values_from = c(estimate, moe),
    names_sep   = "__",
    values_fn   = list(estimate = dplyr::first, moe = dplyr::first)
  )
# Keep only block groups inside the city polygon
acs_wide <- acs_wide %>% filter(GEOID %in% valid_geoids)
# acs_wide is a plain tibble here — no geometry attached yet

# Ethnicity columns for Shannon entropy computation
ethnicity_est_cols <- c(
  "estimate__nh_white",
  "estimate__nh_black",
  "estimate__nh_native",
  "estimate__nh_asian",
  "estimate__nh_pacific",
  "estimate__nh_other",
  "estimate__nh_multiracial",
  "estimate__hispanic"
)

# Compute all derived covariates on the plain tibble (no geometry).
# FIX: running pmap_dbl on a plain tibble prevents sf sticky-geometry from
# injecting polygon coordinates into the entropy function's ... argument,
# which was causing all entropy values to be wrong (and 29 to be negative).
acs_derived <- acs_wide %>%
  mutate(
    percentage_minority = if_else(
      estimate__total_pop > 0,
      (estimate__total_pop - estimate__nh_white) / estimate__total_pop,
      NA_real_
    ),
    # Poverty rate via C17002 (below 0.50 + 0.50–0.99 = below poverty line)
    poverty_rate = if_else(
      estimate__poverty_ratio_total > 0,
      (estimate__poverty_ratio_below_half + estimate__poverty_ratio_half_to_1) /
        estimate__poverty_ratio_total,
      NA_real_
    ),
    unemployment_rate = if_else(
      estimate__labor_force > 0,
      estimate__unemployed / estimate__labor_force,
      NA_real_
    ),
    occupancy_rate = if_else(
      estimate__housing_units > 0,
      estimate__occupied_housing_units / estimate__housing_units,
      NA_real_
    ),
    bachelors_or_higher_rate = if_else(
      estimate__total_education_25_plus > 0,
      (estimate__bachelors + estimate__masters +
         estimate__professional + estimate__doctorate) /
        estimate__total_education_25_plus,
      NA_real_
    ),
    shannon_entropy = pmap_dbl(
      # select() on a plain tibble returns exactly the 8 columns requested —
      # no geometry column injected silently
      select(., all_of(ethnicity_est_cols)),
      function(...) {
        vals  <- as.numeric(unlist(list(...)))
        total <- sum(vals, na.rm = TRUE)
        if (!is.finite(total) || total <= 0) return(NA_real_)
        p <- vals / total
        p <- p[is.finite(p) & p > 0]
        if (length(p) == 0) return(NA_real_)
        -sum(p * log(p))
      }
    )
  )

# Reattach geometry, then reproject to match detection point CRS
acs_sf <- acs_derived %>%
  left_join(
    acs_geo %>% select(GEOID, NAMELSAD = any_of("NAMELSAD"), geometry),
    by = "GEOID"
  ) %>%
  st_as_sf()

acs_projected <- st_transform(acs_sf, target_crs)

# Clean up: remove rows with missing Shannon entropy (zero-population block groups)
# These will be excluded from regression anyway but keep them here for reference
acs_final <- acs_projected %>%
  filter(!is.na(shannon_entropy))

message("Block groups after entropy filter: ", nrow(acs_final),
        " (from ", nrow(acs_projected), " with geometries)")

out_geo <- here::here("data", "philly_combined_revision", "acs_philly_bg_2020_2024.gpkg")
out_csv <- here::here("data", "philly_combined_revision", "acs_philly_bg_2020_2024_derived.csv")

dir.create(dirname(out_geo), recursive = TRUE, showWarnings = FALSE)

message("Writing outputs...")
st_write(acs_final, out_geo, delete_dsn = TRUE, quiet = TRUE)
acs_final %>%
  st_drop_geometry() %>%
  write_csv(out_csv)

message("Done.")
message("GeoPackage : ", out_geo)
message("CSV        : ", out_csv)
message("Rows       : ", nrow(acs_final))
message("Entropy NAs: ", sum(is.na(acs_final$shannon_entropy)))
message("Poverty NAs: ", sum(is.na(acs_final$poverty_rate)))
message("Income NAs : ", sum(is.na(acs_final$estimate__median_household_income)))