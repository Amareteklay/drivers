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

# Count number of unique reports
eppo_joined %>%
  distinct(number, title, date, url) %>%
  nrow() # 6910

eppo_report_country <- import(here(
  "data",
  "eppo_downloads",
  "eppo_reporting_countries.csv"
)) %>%
  clean_names()

# %% Identify merge issues between eppo_joined and eppo_report_country

keys <- c("number", "title", "date", "url")
# Count occurrences in each dataset by join keys
dup_eppo <- eppo_joined %>%
  count(across(all_of(keys)), name = "n_eppo")

dup_country <- eppo_report_country %>%
  count(across(all_of(keys)), name = "n_country")

dup_all <- dup_eppo %>%
  left_join(dup_country, by = keys) %>%
  replace_na(list(n_country = 0))

# ============================================================
# CLASSIFICATION
# ============================================================
one_to_one <- dup_all %>% filter(n_eppo == 1 & n_country == 1)
one_to_zero <- dup_all %>% filter(n_eppo == 1 & n_country == 0)
one_to_many <- dup_all %>% filter(n_eppo == 1 & n_country > 1)
many_to_one <- dup_all %>% filter(n_eppo > 1 & n_country == 1)
many_to_many <- dup_all %>% filter(n_eppo > 1 & n_country > 1)
many_to_zero <- dup_all %>% filter(n_eppo > 1 & n_country == 0)

# ============================================================
# 3) COUNTRY VECTOR PER KEY
# ============================================================
country_list <- eppo_report_country %>%
  group_by(across(all_of(keys))) %>%
  summarise(
    country_vector = paste(sort(unique(country)), collapse = "; "),
    .groups = "drop"
  )

# ----  ONE-TO-ONE  ----
eppo_country_one_one <- eppo_joined %>%
  inner_join(one_to_one %>% select(all_of(keys)), by = keys) %>%
  left_join(country_list, by = keys) %>%
  mutate(merge_case = "one_to_one")

# ----  ONE-TO-ZERO  ----
eppo_country_one_zero <- eppo_joined %>%
  inner_join(one_to_zero %>% select(all_of(keys)), by = keys) %>%
  mutate(country_vector = NA_character_, merge_case = "one_to_zero")

# ----  ONE-TO-MANY  ----
eppo_country_one_many <- eppo_joined %>%
  inner_join(one_to_many %>% select(all_of(keys)), by = keys) %>%
  left_join(country_list, by = keys) %>%
  mutate(merge_case = "one_to_many")

# ----  MANY-TO-ONE  ----
eppo_country_many_one <- eppo_joined %>%
  inner_join(many_to_one %>% select(all_of(keys)), by = keys) %>%
  left_join(country_list, by = keys) %>%
  mutate(merge_case = "many_to_one")

# ----  MANY-TO-MANY  ----
eppo_country_many_many <- eppo_joined %>%
  inner_join(many_to_many %>% select(all_of(keys)), by = keys) %>%
  left_join(country_list, by = keys) %>%
  mutate(merge_case = "many_to_many")

# ----  MANY-TO-ZERO  ----
eppo_country_many_zero <- eppo_joined %>%
  inner_join(many_to_zero, by = keys) %>%
  mutate(country_vector = NA_character_, merge_case = "many_to_zero")

eppo_country_final <- bind_rows(
  eppo_country_one_one,
  eppo_country_one_zero,
  eppo_country_one_many,
  eppo_country_many_one,
  eppo_country_many_many,
  eppo_country_many_zero
) %>%
  select(-n_eppo, -n_country, country = country_vector)

export(
  eppo_country_final,
  here("data", "eppo_downloads", "eppo_occurence_temp.csv")
)

# LAPHFR fall armyworm example dataset
eppo_laphfr <- eppo_country_final %>%
  filter(eppo_code == "LAPHFR") %>%
  # move content column to the end
  relocate(content, .after = last_col()) %>%
  mutate(report_year = ymd(report_year)) %>%
  arrange(report_year)

export(
  eppo_laphfr,
  here("data", "eppo_downloads", "eppo_laphfr.csv")
)

# #TODO: Filter irrelevant reports later
# eppo_unmerged_filtered <- eppo_unmerged %>%
#   # The following reports missing countries, but can drop completely
#   filter(
#     !str_detect(
#       title,
#       "Recent updates in the EPPO Global Database|Pests which should not appear in EPPO Quarantine lists|Update of the list of invasive alien species|New EU Regulations|New and revised dynamic EPPO datasheets are available in the EPPO Global Database|Changes made to the EU list of regulated pests|New additions to the EPPO Lists|New additions to the EPPO A1 and A2 Lists|New EU regulation|Guidelines for the management|A1 and A2 list|Prioritization of invasive alien plants|Prioritization of alien plants|New EPPO lists of invasive alien plants|Q-bank database on invasive alien plants|Binomial nomenclature for virus species|Invasive Alien Plants in European Macaronesia|Recognition and management guides for invasive alien plants in Belgium"
#     )
#   )

# New analysis -----------------------------------------------------------
eppo_expanded <- eppo_country_final %>%
  separate_rows(country, sep = ";\\s*") %>%
  filter(!is.na(country), country != "")


# fmt: skip

# Archived code ----------------------------------------------------------

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
yearly_report_counts <- eppo_joined %>%
  mutate(year = year(report_year)) %>%
  count(year, name = "reports") %>%
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
first_report <- eppo_joined %>%
  group_by(eppo_code, name) %>%
  summarise(first_report_year = min(year(report_year)), .groups = 'drop') %>%
  arrange(first_report_year)

# Count new pathogens appearing each year
new_pathogens_by_year <- first_report %>%
  count(first_report_year, name = "new_pathogens") %>%
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
