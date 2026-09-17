# Data sources and attribution

## Field water quality

USGS and the Water Quality Portal provide the field observations used to construct lake–day chlorophyll-a values and observed events. The study retains observations from 2005–2025; external event evaluation uses 2021–2025. Source extracts and cleaned records are in `04_analysis_data/input_data/`. External lake–day records, quality-control summaries and event tables are in `04_analysis_data/primary_external_evaluation/event_outcomes/`.

- USGS Water Data for the Nation: https://doi.org/10.5066/F7P55KJN
- USGS Water Data: https://waterdata.usgs.gov/
- Water Quality Portal: https://www.waterqualitydata.us/
- Read, E.K., Carr, L., De Cicco, L., Dugan, H.A., Hanson, P.C., Hart, J.A., Kreft, J., Read, J.S., Winslow, L.A., 2017. Water quality data for national-scale aquatic research: The Water Quality Portal. Water Resources Research 53, 1735–1745. https://doi.org/10.1002/2016WR019993

The historical water-quality extract identifies LAGOS-US LIMNO data package `edi.1439.5` (Shuvo et al., 2023): https://doi.org/10.6073/pasta/2c58f5a50ab813919f99cc1f265f271c.

## Lake identities and attributes

Lake identifiers and spatial attributes were organized using LAGOS and AquaSat-derived records. The study lake list and spatial support tables are in `06_code/figure_generation/resources/geography/`; lake–prediction-date records are in `04_analysis_data/primary_external_evaluation/candidate_features/`.

- Cheruvelil, K.S., Soranno, P.A., McCullough, I.M., Webster, K.E., Rodriguez, L.K., Smith, N.J., 2021. LAGOS-US LOCUS v1.0: Data module of location, identifiers, and physical characteristics of lakes and their watersheds in the conterminous U.S. Limnology and Oceanography Letters 6, 270–292. https://doi.org/10.1002/lol2.10203
- Ross, M.R.V., Topp, S.N., Appling, A.P., Yang, X., Kuhn, C., Butman, D., Simard, M., Pavelsky, T.M., 2019. AquaSat: A Data Set to Enable Remote Sensing of Water Quality for Inland Waters. Water Resources Research 55, 10012–10025. https://doi.org/10.1029/2019WR024883

## Meteorology

Daily meteorological inputs are derived from ERA5 hourly time-series data on single levels (`reanalysis-era5-single-levels-timeseries`), as identified by the source fields in the stored daily table. The local daily table and lake-to-grid mapping are in `04_analysis_data/input_data/`. External predictor tables and the predictor dictionary are in `04_analysis_data/primary_external_evaluation/meteorology_features/`.

- Copernicus Climate Change Service (2025). ERA5 hourly time-series data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/1cf1ad76
- Copernicus Climate Data Store: https://cds.climate.copernicus.eu/
- Hersbach, H., et al., 2020. The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society 146, 1999–2049. https://doi.org/10.1002/qj.3803

Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

## Satellite observations

Satellite predictors are derived from the EPA Cyanobacteria Assessment Network (CyAN) lake-level observation product. Lake–week history, duplicate-resolution records and external predictors are in `04_analysis_data/primary_external_evaluation/satellite_features/`.

- EPA CyAN: https://www.epa.gov/water-research/cyanobacteria-assessment-network-cyan
- EPA CyAN application and data-service documentation: https://www.epa.gov/water-research/cyanobacteria-assessment-network-application-cyan-app

## Geographic context

The map renderer uses a cached Natural Earth II base map and the included 2023 state-boundary GeoJSON. These provide geographic context for the study lake coordinates. Natural Earth data are in the public domain: https://www.naturalearthdata.com/about/terms-of-use/.

## File identity and reuse

The archive manifests identify the distributed files. `provenance/evaluation_inputs.json` records the exact numerical inputs used by the evaluation. The files here are the study's source extracts and derived analysis tables, rather than complete copies of the providers' databases. Third-party data remain subject to their original provider terms and attribution requirements.
