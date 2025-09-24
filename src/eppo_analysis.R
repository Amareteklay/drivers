# Load packages and data -------------------------------------------------

# %% Load package
pacman::p_load(
  rio,
  here,
  janitor,
  rnaturalearth,
  sf,
  gghighlight,
  tidyverse
)

# %% Import data
eppo_occurrence <- import(here(
  "data",
  "eppo_downloads",
  "eppo_reporting_occurrence_LLM.csv"
)) %>%
  clean_names()

eppo_occurrence_analysis <- eppo_occurrence |>
  filter(
    type_of_record %in%
      c(
        "First record",
        "Update of outbreak",
        "Eradication of disease",
        "Absence of disease"
      )
  ) |>
  mutate(
    report_year = lubridate::ym(date),
    #NOTE Create year1: standardized date from the 'year' column
    #NOTE The year column contains messy data (ranges, "unknown", etc.)
    #NOTE This converts it to a consistent Date format using the first year mentioned
    first_year_occurence = {
      y <- stringr::str_squish(as.character(year))
      dplyr::case_when(
        is.na(y) ~ NA_Date_,
        stringr::str_to_lower(y) == "unknown" ~ NA_Date_,
        stringr::str_detect(y, "^[0-9]{4}$") ~ readr::parse_date(y, "%Y"),
        stringr::str_detect(y, "^[0-9]{4}\\s*[-–]\\s*[0-9]{4}$") ~
          readr::parse_date(stringr::str_extract(y, "^[0-9]{4}"), "%Y"),
        TRUE ~ readr::parse_date(stringr::str_extract(y, "[0-9]{4}"), "%Y")
      )
    },
    # Lag between first year of occurrence and report date in term of months
    lag_months = as.numeric(
      lubridate::interval(first_year_occurence, report_year) /
        lubridate::duration(months = 1)
    )
  )
# %% Top 10 diseases/pests by total mentions
eppo_occurrence_analysis |>
  count(name, sort = TRUE, name = "total_records")

# describe the lag
summary(eppo_occurrence_analysis$lag_months)

# PLOTS --------------------------------------------------------------
# %% Overall trend per year
overall_year <- eppo_occurrence_analysis |>
  count(first_year_occurence)

ggplot(overall_year, aes(x = first_year_occurence, y = n)) +
  geom_col(fill = "#3E7CB1") +
  geom_line(aes(group = 1), color = "#114B5F", linewidth = 0.8) +
  geom_smooth(method = "loess", se = FALSE, color = "#E76F51", span = 0.8) +
  labs(
    title = "EPPO occurrences over time (all pests/diseases)",
    x = "Occurence year",
    y = "Number of occurrences"
  ) +
  theme_minimal(base_size = 12)

# %% Overall trend per year - per continent
continent_year <- eppo_occurrence_analysis |>
  count(continent, first_year_occurence)

# Occurrences by continent over time (stacked area)
ggplot(continent_year, aes(x = first_year_occurence, y = n, fill = continent)) +
  geom_area(alpha = 0.85) +
  labs(
    title = "EPPO occurrences over time by continent",
    x = "Occurence year",
    y = "Number of occurrences"
  ) +
  scale_x_date(
    expand = c(0, 0),
    date_breaks = "5 years",
    date_minor_breaks = "5 years",
    label = scales::label_date_short()
  ) +
  ggsci::scale_fill_bmj() +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

# other way to highlight using gghighlight package
eppo_occurrence_analysis |>
  drop_na(first_year_occurence) |>
  ggplot() +
  geom_histogram(
    aes(x = first_year_occurence, group = continent, fill = continent),
    color = "black"
  ) +
  gghighlight() +
  facet_wrap(~continent) +
  scale_x_date(
    expand = c(0, 0),
    date_breaks = "10 years",
    date_minor_breaks = "5 years",
    label = scales::label_date_short()
  ) +
  labs(
    title = "EPPO occurrences over time by continent",
    x = "Occurence year",
    y = "Number of occurrences"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    plot.caption = element_text(face = "italic", hjust = 0), # caption on left side in italics
    axis.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 10),
    strip.background = element_rect(fill = "white")
  )

# %% Top 3 diseases/pests occurrences over time by continent
eppo_occurrence_analysis |>
  drop_na(first_year_occurence) |>
  ggplot() +
  geom_histogram(
    aes(x = first_year_occurence, group = name, fill = name),
    color = "black"
  ) +
  gghighlight() +
  facet_grid(continent ~ name) +
  scale_x_date(
    expand = c(0, 0),
    date_breaks = "10 years",
    date_minor_breaks = "5 years",
    label = scales::label_date_short()
  ) +
  labs(
    title = "Top 3 EPPO occurrences over time by continent",
    x = "Occurence year",
    y = "Number of occurrences"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    plot.caption = element_text(face = "italic", hjust = 0), # caption on left side in italics
    axis.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 10),
    strip.background = element_rect(fill = "white")
  )

# %% The number of reports by country - world map
report_per_country <- eppo_occurrence_analysis |>
  count(country, sort = TRUE, name = "total_reports")

world <- rnaturalearth::ne_countries(scale = "medium", returnclass = "sf")
world_report <- left_join(
  world,
  report_per_country,
  by = c("admin" = "country")
)

# Create world map of reports by country
ggplot(world_report) +
  geom_sf(aes(fill = total_reports), color = "white", size = 0.1) +
  scale_fill_viridis_c(
    name = "Total Reports",
    na.value = "grey90",
    breaks = c(1, 5, 10, 20, 50),
    labels = c("1", "5", "10", "20", "50+")
  ) +
  labs(
    title = "EPPO Disease/Pest Reports by Country",
    subtitle = "Number of occurrence reports submitted to EPPO",
    caption = "Grey areas indicate no reports or missing data"
  ) +
  theme_void(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(1.5, "cm"),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic", hjust = 0)
  )

# %% The biogeography of disease outbreaks.
# As a final illustration of how these data could be used for more quantitative research, we conducted a simple analysis to explore which countries report more similar kinds of diseases. To do so, we counted the number of times a given disease was reported in each country, root-transformed this matrix (given how many updates have been filed in a small number of epidemics), and sent this matrix through a k-means clustering algorithm (with an arbitrarily chosen k = 7 clusters)

# %% The biogeography of disease outbreaks - limit to top 3 diseases for readability
# Get top 3 diseases
top_diseases <- eppo_occurrence_analysis |>
  count(name, sort = TRUE) |>
  slice_head(n = 3) |>
  pull(name)

# Create report counts for top diseases only
report_per_country <- eppo_occurrence_analysis |>
  filter(name %in% top_diseases) |>
  count(country, disease = name, sort = TRUE, name = "total_reports")

world <- rnaturalearth::ne_countries(scale = "medium", returnclass = "sf")

# Create complete combinations of world countries and diseases
world_expanded <- map_dfr(
  top_diseases,
  ~ {
    world |>
      mutate(disease = .x)
  }
) |>
  left_join(report_per_country, by = c("admin" = "country", "disease"))

# Create world map of reports by country for top diseases
ggplot(world_expanded) +
  geom_sf(aes(fill = total_reports), color = "white", size = 0.1) +
  scale_fill_viridis_c(
    name = "Total Reports",
    na.value = "grey90",
    breaks = c(1, 5, 10, 20, 50),
    labels = c("1", "5", "10", "20", "50+")
  ) +
  facet_wrap(~disease, ncol = 2) +
  labs(
    title = "EPPO Disease/Pest Reports by Country - Top 3 Diseases",
    subtitle = "Number of occurrence reports submitted to EPPO",
    caption = "Grey areas indicate no reports or missing data"
  ) +
  theme_void(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(1.5, "cm"),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic", hjust = 0),
    strip.text = element_text(face = "bold", size = 10)
  )
