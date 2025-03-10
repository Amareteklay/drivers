# Define few-shot examples as JSON-serializable dictionaries
trainset = [
    {
        "text": "High population density and mobility in urban areas have facilitated the rapid spread of the virus.",
        "cause": "High population density and mobility in urban areas",
        "effect": "the rapid spread of the virus"
    },
    {
        "text": ("There is no vaccine for Influenza A(H1N1)v infection currently licensed for use in humans. "
                 "Seasonal influenza vaccines against human influenza viruses are generally not expected to protect people "
                 "from infection with influenza viruses that normally circulate in pigs, but they can reduce severity."),
        "cause": "no vaccine for Influenza A(H1N1)v infection currently licensed for use in humans",
        "effect": "infection with influenza viruses that normally circulate in pigs"
    },
    {
        "text": ("Several countries including Cameroon, Ethiopia, Haiti, Lebanon, Nigeria, Pakistan, Somalia, Syria, and the "
                 "Democratic Republic of Congo are in the midst of complex humanitarian crises with fragile health systems, "
                 "inadequate access to clean water and sanitation, and insufficient capacity to respond to the outbreaks."),
        "cause": ("complex humanitarian crises with fragile health systems, inadequate access to clean water and sanitation, "
                  "and insufficient capacity"),
        "effect": "insufficient capacity to respond to the outbreaks"
    },
    {
        "text": ("Moreover, a low index of suspicion, socio-cultural norms, community resistance, limited community knowledge "
                 "regarding anthrax transmission, high levels of poverty and food insecurity, a shortage of available vaccines "
                 "and laboratory reagents, inadequate carcass disposal and decontamination practices significantly contribute to "
                 "hampering the containment of the anthrax outbreak."),
        "cause": ("low index of suspicion, socio-cultural norms, community resistance, limited community knowledge regarding "
                  "anthrax transmission, high levels of poverty and food insecurity, shortage of available vaccines and laboratory reagents, "
                  "inadequate carcass disposal and decontamination practices"),
        "effect": "hampering the containment of the anthrax outbreak"
    },
    {
        "text": ("The risk of insufficient control capacities is considered high in Zambia due to concurrent public health emergencies "
                 "in the country (cholera, measles, COVID-19) that limit the country’s human and financial capacities to respond to the current "
                 "anthrax outbreak adequately."),
        "cause": "concurrent public health emergencies in the country (cholera, measles, COVID-19)",
        "effect": "limit the country’s human and financial capacities to respond to the current anthrax outbreak"
    },
    {
        "text": ("Surveillance systems specifically targeting endemic transmission of chikungunya or Zika are weak or non-existent, leading to "
                 "misdiagnosis between diseases & skewed surveillance, which in turn misinforms policy decisions and reduces the accuracy on the "
                 "estimation of the true burden of each disease."),
        "cause": "weak or non-existent surveillance systems",
        "effect": ("misdiagnosis between diseases, skewed surveillance, misinformed policy decisions, and reduced accuracy on the estimation "
                   "of the true burden of each disease")
    },
    {
        "text": ("Changes in the predominant circulating serotype increase the population risk of subsequent exposure to a heterologous DENV serotype, "
                 "which increases the risk of higher rates of severe dengue and deaths."),
        "cause": "changes in the predominant circulating serotype",
        "effect": "increased risk of severe dengue and deaths"
    }
]