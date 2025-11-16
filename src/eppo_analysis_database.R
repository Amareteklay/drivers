# Load packages and data -------------------------------------------------

# %% Load package
pacman::p_load(
  rio,
  here,
  janitor,
  tidyverse
)

# %% Import eppo_code data
eppo_code <- import(here(
  "data",
  "eppo_downloads",
  "eppo_code.csv"
)) %>%
  clean_names()

# %% Import crawled data
eppo_files <- list.files(
  path = here("data", "eppo_downloads"),
  pattern = "eppo_reporting_.*\\k.csv$",
  full.names = TRUE
)

# Reorder 105k and 126k to the end of the list
eppo_files <- c(eppo_files[3:length(eppo_files)], eppo_files[1:2])

eppo_joined <- map_dfr(eppo_files, import, .id = "source_file") %>%
  clean_names() %>%
  mutate(
    source_file = basename(eppo_files[as.numeric(source_file)]),
    report_year = lubridate::ym(date),
    # replace NA for column report_year to 1987-01
    report_year = if_else(
      is.na(report_year),
      lubridate::ym("1987-01"),
      report_year
    )
  )
export(
  eppo_joined,
  here("data", "eppo_downloads", "eppo_crawled_joined.csv")
)

# Crawl performance review -----------------------------------------------

# Check hown many eppo codes are missing in the eppo_code database
#NOTE 121837 eppo_code have no reports
eppo_missing_codes <- eppo_code %>%
  anti_join(
    eppo_joined %>%
      distinct(eppo_code),
    by = c("eppocode" = "eppo_code")
  )

100 - (nrow(eppo_missing_codes) / nrow(eppo_code) * 100) # 3.376%

# count number of unique eppo_codes in the joined data
eppo_joined %>%
  distinct(eppo_code) %>%
  nrow()


# Count number of reports by eppo_code
eppo_report_count <- eppo_joined %>%
  group_by(eppo_code) %>%
  summarise(report_count = n(), .groups = "drop") %>%
  arrange(desc(report_count))

summary(eppo_report_count$report_count)

# Histogram of report counts
ggplot(eppo_report_count, aes(x = report_count)) +
  geom_histogram(binwidth = 1, color = "seagreen") +
  labs(title = "Distribution of Report Counts by EPPO Code") +
  xlab("Number of Reports") +
  ylab("Frequency") +
  theme_minimal() +
  # increase font size
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    # axis text size
    axis.text.x = element_text(size = 14),
    axis.text.y = element_text(size = 14),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14)
  )

# Exploring data ---------------------------------------------------------

# %% Distribution of reports by year
# Time distribution of reports
yearly_report_counts <- eppo_joined |>
  mutate(year = year(report_year)) |>
  count(year, name = "reports") |>
  arrange(year)


ggplot(yearly_report_counts, aes(x = year, y = reports)) +
  geom_line(color = "#FF5733", linewidth = 1) +
  geom_point(color = "#FF5733", size = 2) +
  labs(
    title = "EPPO Plant Disease Reports Over Time",
    subtitle = "Number of reports by year (1974-2025)",
    x = "Year",
    y = "Number of Reports"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12)
  )

ggplot(yearly_report_counts, aes(x = year, y = reports)) +
  geom_col(fill = "seagreen", color = "white") +
  labs(
    title = "EPPO Plant Disease Reports Over Time",
    subtitle = "Number of reports by year (1974-2025)",
    x = "Year",
    y = "Number of reports"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12)
  )

# return the aggregated table as the final object
yearly_report_counts

# %% Timeline of new pathogen emergence by year
# Find the first report year for each unique pathogen (eppo_code)
first_report <- eppo_joined |>
  group_by(eppo_code, name) |>
  summarise(first_report_year = min(year(report_year)), .groups = 'drop') |>
  arrange(first_report_year)

# Count new pathogens appearing each year
new_pathogens_by_year <- first_report |>
  count(first_report_year, name = "new_pathogens") |>
  arrange(first_report_year)

# Plot new pathogens by year
ggplot(new_pathogens_by_year, aes(x = first_report_year, y = new_pathogens)) +
  geom_line(color = "#FF5733", linewidth = 1) +
  geom_point(color = "#FF5733", size = 2) +
  labs(
    title = "New Pathogen Emergence Over Time",
    subtitle = "Number of new pathogens reported each year",
    x = "Year",
    y = "Number of New Pathogens"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12)
  )

ggplot(new_pathogens_by_year, aes(x = first_report_year, y = new_pathogens)) +
  geom_col(fill = "seagreen", color = "white") +
  labs(
    title = "New Pathogen Emergence Over Time",
    subtitle = "Number of new pathogens reported each year",
    x = "Year",
    y = "Number of New Pathogens"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12)
  )

# %% Geographic distribution of reports

# %% Analysis 16 November
