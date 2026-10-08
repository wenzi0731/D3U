# HEEW data schema

Data files are not tracked. Use the same cleaned HEEW files as `2-stages`.

`CN03_energy_cleaned.csv`:

```text
Year, Month, Day, Hour, Electricity, Heat, Cooling, PV
```

`weather_cleaned.csv` for the default `pv10` setting:

```text
Year, Month, Day, Hour, Temperature, Dew Point, Humidity, Wind Speed,
Pressure, Precip, ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN,
PV_CLEARNESS_RATIO, PV_IS_DAYLIGHT
```

Timestamps must be unique. Only aligned complete 24-hour days are retained.

