# Published data schema

Wind Farm A and Wind Farm B ECMWF files are author-prepared forecast archives aligned to target valid times. They do not record forecast issue or availability times. See manuscript Section 3.1 and Supplementary Section S3.4 for the operational schedule.

Kelmarsh GFS contains 367 daily 00 UTC runs from 2023-12-31 through 2024-12-31, leads 3–33 h. Availability times come from the original S3 LastModified; the loader chooses the latest run available at each origin.

SCADA power columns use kW; wind-speed columns use m/s. Wind Farm A Total_Power uses MW. Timestamp strings and all retained observation and forecast values are unchanged. Missing values retain the source representation. The loader preserves raw valid targets and uses forward-only history filling.

A/B turbine x_coord, y_coord and elevation_m are relative metres with per-axis minima zero. Float32 rounding follows the frozen loader before translation. A turbine order is T01–T29; B is T01–T72. Kelmarsh uses IDs 1–6. The model uses pairwise relative positions and elevation differences. Frozen normalisation statistics accompany each checkpoint; use the recorded chronological splits in its metrics configuration.

Wind Farm A terrain inputs and all DEM-derived geometry are withheld to preserve site anonymity. Published CFD lookup tables, intersector checks, wake and roughness sensitivity results, and quality/recovery fields remain available.

Wind Farm A includes 11 ECMWF fields (9 model features plus 100 m u/v controls); 11 unused fields were removed. Wind Farm B retains all 22 ECMWF fields. nwp_geopotential values and every associated frozen statistic are unchanged. Radiation, water-vapour and geopotential fields retain archive units where no independent unit verification was available.

## Wind Farm A ECMWF columns

- `nwp_direct_solar_radiation`
- `nwp_total_sky_direct_shortwave`
- `nwp_surface_pressure_pa`
- `nwp_temperature_2m_k`
- `nwp_u10_ms`
- `nwp_v10_ms`
- `nwp_u100_ms`
- `nwp_v100_ms`
- `nwp_wind_speed10_ms`
- `nwp_wind_speed200_ms`
- `nwp_wind_speed100_ms`

## Wind Farm B ECMWF columns

- `nwp_u100_ms`
- `nwp_v100_ms`
- `nwp_u10_ms`
- `nwp_v10_ms`
- `nwp_u200_ms`
- `nwp_v200_ms`
- `nwp_dewpoint_2m_k`
- `nwp_temperature_2m_k`
- `nwp_direct_solar_radiation`
- `nwp_geopotential`
- `nwp_high_cloud_cover`
- `nwp_mean_sea_level_pressure_pa`
- `nwp_skin_temperature_k`
- `nwp_surface_pressure_pa`
- `nwp_surface_solar_radiation_clear_sky`
- `nwp_surface_shortwave_downwards`
- `nwp_total_cloud_cover`
- `nwp_total_column_water_vapour`
- `nwp_toa_incident_solar_radiation`
- `nwp_wind_speed100_ms`
- `nwp_wind_speed10_ms`
- `nwp_wind_speed200_ms`
