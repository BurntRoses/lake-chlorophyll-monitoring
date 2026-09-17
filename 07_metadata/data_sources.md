# Data sources and attribution

## Field water quality

USGS Water Data for the Nation and the Water Quality Portal provide the field observations used to construct lake–day chlorophyll-a values and observed events. The study retains observations from 2005–2025; external event evaluation uses 2021–2025. Source extracts and cleaned records are distributed in the analysis-data release. External lake–day records, quality-control summaries and event tables are in the restored `04_analysis_data/primary_external_evaluation/event_outcomes/` tree.

- U.S. Geological Survey. *USGS Water Data for the Nation: U.S. Geological Survey National Water Information System database*. https://doi.org/10.5066/F7P55KJN
- USGS Water Data: https://waterdata.usgs.gov/
- Water Quality Portal: https://www.waterqualitydata.us/
- Read, E.K., Carr, L., De Cicco, L., Dugan, H.A., Hanson, P.C., Hart, J.A., Kreft, J., Read, J.S., Winslow, L.A., 2017. Water quality data for national-scale aquatic research: The Water Quality Portal. *Water Resources Research* 53, 1735–1745. https://doi.org/10.1002/2016WR019993

The historical water-quality extract identifies LAGOS-US LIMNO data package `edi.1439.5` (Shuvo et al., 2023): https://doi.org/10.6073/pasta/2c58f5a50ab813919f99cc1f265f271c.

## Lake identities and attributes

Lake identifiers and spatial attributes were organized using LAGOS-US LOCUS and AquaSat-derived records. The study lake list and spatial support tables are on the default branch under `06_code/figure_generation/resources/geography/`; lake–prediction-date records are in the release-only analysis-data tree.

- Cheruvelil, K.S., Soranno, P.A., McCullough, I.M., Webster, K.E., Rodriguez, L.K., Smith, N.J., 2021. LAGOS-US LOCUS v1.0: Data module of location, identifiers, and physical characteristics of lakes and their watersheds in the conterminous U.S. *Limnology and Oceanography Letters* 6, 270–292. https://doi.org/10.1002/lol2.10203
- Ross, M.R.V., Topp, S.N., Appling, A.P., Yang, X., Kuhn, C., Butman, D., Simard, M., Pavelsky, T.M., 2019. AquaSat: A Data Set to Enable Remote Sensing of Water Quality for Inland Waters. *Water Resources Research* 55, 10012–10025. https://doi.org/10.1029/2019WR024883

## Meteorology

Daily meteorological inputs are derived from ERA5 hourly time-series data on single levels (`reanalysis-era5-single-levels-timeseries`), as identified by the source fields in the stored daily table. The local daily table, lake-to-grid mapping and external meteorological predictors are distributed in the analysis-data release.

- Copernicus Climate Change Service, 2025. *ERA5 hourly time-series data on single levels from 1940 to present*. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/1cf1ad76
- Copernicus Climate Data Store: https://cds.climate.copernicus.eu/
- Hersbach, H., Bell, B., Berrisford, P., et al., 2020. The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society* 146, 1999–2049. https://doi.org/10.1002/qj.3803

Required attribution used in the manuscript: "This study contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains."

## Satellite cyanobacteria observations

Satellite predictors are derived from the U.S. Environmental Protection Agency Cyanobacteria Assessment Network (CyAN) products based on Copernicus Sentinel-3 Ocean and Land Colour Instrument (OLCI) observations. EPA describes CyAN/CyANWeb as providing satellite-derived cyanobacteria measures for larger U.S. lakes and reservoirs, including 7-day maximum products updated weekly and daily snapshots for later periods. The study's lake-level cyanobacteria histories, duplicate-resolution records and external predictors are distributed in the restored `04_analysis_data/primary_external_evaluation/satellite_features/` tree.

Primary documentation and citation:

- U.S. Environmental Protection Agency. Cyanobacteria Assessment Network (CyAN): https://www.epa.gov/water-research/cyanobacteria-assessment-network-cyan
- U.S. Environmental Protection Agency. Cyanobacteria Assessment Network Application / CyANWeb documentation: https://www.epa.gov/water-research/cyanobacteria-assessment-network-application-cyan-app
- Schaeffer, B.A., Bailey, S.W., Conmy, R.N., Galvin, M., Ignatius, A.R., Johnston, J.M., Keith, D.J., Lunetta, R.S., Parmar, R., Stumpf, R.P., Urquhart, E.A., Werdell, P.J., Wolfe, K., 2018. Mobile device application for monitoring cyanobacteria harmful algal blooms using Sentinel-3 satellite Ocean and Land Colour Instruments. *Environmental Modelling & Software* 109, 93–103. https://doi.org/10.1016/j.envsoft.2018.08.015

The 2018 paper documents the Sentinel-3 OLCI basis of the CyAN application and its use for cyanobacteria monitoring; the current EPA pages document the operational CyAN/CyANWeb service and product availability. The repository does not treat the 2024 forecasting paper as the source citation for the observational CyAN product.

## Geographic context

The map renderer uses a cached Natural Earth II base map and an included 2023 state-boundary GeoJSON to provide geographic context for study lake coordinates. Natural Earth data are in the public domain: https://www.naturalearthdata.com/about/terms-of-use/.

## File identity and reuse

The archive manifests identify the distributed files. `07_metadata/provenance/evaluation_inputs.json` records the exact numerical input identities used by the evaluation. Files distributed with this study are source extracts and derived analysis tables rather than complete mirrors of provider databases. Third-party data remain subject to their original provider terms and attribution requirements.