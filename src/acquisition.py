from __future__ import annotations
import argparse
import csv
import hashlib
import html
import re
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

PACKAGE = {
    "scope": "edi",
    "identifier": "1439",
    "revision": "5",
    "package_id": "edi.1439.5",
    "doi": "10.6073/pasta/2c58f5a50ab813919f99cc1f265f271c",
    "metadata_url": "https://pasta.lternet.edu/package/metadata/eml/edi/1439/5",
}

CHEMISTRY_ENTITY = "chemistry_limno.csv"

SOURCE_ENTITY = "sourcedata_limno.csv"

CHEMISTRY_COLUMNS = [
    "eventid_epi",
    "lagoslakeid",
    "sample_date",
    "sample_id",
    "parameter_id",
    "parameter_name",
    "parameter_value",
    "parameter_detectionlimit_value",
    "censorcode",
    "sample_depth_m",
    "sample_depth_flag",
    "cluster_id",
    "cluster_lat_dd",
    "cluster_lon_dd",
    "source_id",
    "source_sample_siteid",
]

SOURCE_COLUMNS = [
    "lagoslakeid",
    "sample_id",
    "parameter_id",
    "source_id",
    "source_name",
    "source_activityorg_name",
    "source_sample_siteid",
    "source_activityid",
    "source_comments",
    "source_parameter_name",
    "source_parameter_value",
    "source_parameter_unit",
    "source_value_qualifiercode",
    "parameter_conversionfactor",
    "source_detectionlimit_value",
    "source_detectionlimit_unit",
    "parameter_detectionlimit_conversionfactor",
    "source_detectionlimit_condition",
]

@dataclass
class Entity:
    entity_name: str
    object_name: str
    url: str
    size_bytes: int | None
    md5: str
    description: str

STANDARD_UNITS = {
    "chla": "ug/L",
    "tn": "ug/L",
    "tp": "ug/L",
    "water_temp": "degC",
    "do": "mg/L",
    "ph": "pH",
    "conductivity": "uS/cm",
    "turbidity": "NTU",
    "secchi_depth": "m",
}

CANONICAL_LABELS = {
    "chla": "Chl-a",
    "tn": "TN",
    "tp": "TP",
    "water_temp": "water temperature",
    "do": "dissolved oxygen",
    "ph": "pH",
    "conductivity": "conductivity",
    "turbidity": "turbidity",
    "secchi_depth": "Secchi depth",
}

def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8")

def parse_entities(eml_text: str) -> list[Entity]:
    root = ET.fromstring(eml_text.encode("utf-8"))
    entities: list[Entity] = []
    for table in root.findall(".//dataTable"):
        object_name = table.findtext(".//physical/objectName") or ""
        size_text = table.findtext(".//physical/size")
        md5 = table.findtext(".//physical/authentication") or ""
        entity = Entity(
            entity_name=table.findtext("entityName") or "",
            object_name=object_name,
            url=table.findtext(".//online/url") or "",
            size_bytes=int(size_text) if size_text and size_text.isdigit() else None,
            md5=md5.strip(),
            description=(table.findtext("entityDescription") or "").strip(),
        )
        entities.append(entity)
    return entities

def download_file(entity: Entity, out_dir: Path, force: bool = False) -> Path:
    out_path = out_dir / entity.object_name
    if out_path.exists() and not force:
        if entity.size_bytes is None or out_path.stat().st_size == entity.size_bytes:
            return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with urllib.request.urlopen(entity.url, timeout=180) as response, tmp_path.open("wb") as handle:
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024 * 8)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded and downloaded % (1024 * 1024 * 128) < len(chunk):
                print(f"downloaded {entity.object_name}: {downloaded / (1024 ** 2):.1f} MB")
    tmp_path.replace(out_path)
    return out_path

def normalize_parameter_name(name: object) -> str:
    if pd.isna(name):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(name).strip().lower()).strip()

def canonical_variable(name: object) -> str | None:
    norm = normalize_parameter_name(name)
    compact = norm.replace(" ", "")
    if not norm:
        return None

    if (
        re.search(r"(chlorophyll|chloro|chl\\s*a|^chla$|^chl$)", norm)
        or compact in {"chlorophylla", "chla"}
        or compact.startswith("chla")
    ):
        return "chla"
    if compact in {"tn", "totalnitrogen"} or compact.startswith("tn") or re.search(
        r"total.*nitrogen|nitrogen.*total", norm
    ):
        return "tn"
    if compact in {"tp", "totalphosphorus"} or compact.startswith("tp") or re.search(
        r"total.*phosphorus|phosphorus.*total", norm
    ):
        return "tp"
    if compact in {"temp", "temperature", "watertemp", "watertemperature"} or re.search(
        r"water.*temp|temp.*water|temperature", norm
    ):
        return "water_temp"
    if compact in {"do", "domgl", "dissolvedoxygen"} or re.search(
        r"^do\\b|dissolved.*oxygen|oxygen.*dissolved", norm
    ):
        return "do"
    if compact == "ph" or norm == "p h":
        return "ph"
    if compact.startswith("spcond") or re.search(r"conduct|specific conductance", norm):
        return "conductivity"
    if compact.startswith("turb") or "turbid" in norm:
        return "turbidity"
    if "secchi" in norm:
        return "secchi_depth"
    return None

def read_filtered_chemistry(path: Path, chunksize: int) -> tuple[pd.DataFrame, Counter[str]]:
    chunks: list[pd.DataFrame] = []
    parameter_counts: Counter[str] = Counter()
    for chunk in pd.read_csv(path, usecols=CHEMISTRY_COLUMNS, dtype="string", chunksize=chunksize):
        parameter_counts.update(chunk["parameter_name"].dropna().astype(str))
        chunk["canonical_variable"] = chunk["parameter_name"].map(canonical_variable)
        filtered = chunk.loc[chunk["canonical_variable"].notna()].copy()
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        raise RuntimeError("No target LIMNO parameters matched the B1 variable rules.")
    return pd.concat(chunks, ignore_index=True), parameter_counts

def read_filtered_source(path: Path, sample_ids: set[str], chunksize: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=SOURCE_COLUMNS, dtype="string", chunksize=chunksize):
        filtered = chunk.loc[chunk["sample_id"].isin(sample_ids)].copy()
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        raise RuntimeError("No source rows matched target LIMNO sample IDs.")
    return pd.concat(chunks, ignore_index=True)

def standardize_table(chemistry: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    source = source.drop_duplicates(subset=["sample_id"], keep="first")
    merged = chemistry.merge(
        source,
        on=["sample_id"],
        how="left",
        suffixes=("", "_source"),
        validate="many_to_one",
    )
    output = pd.DataFrame(
        {
            "lake_id": merged["lagoslakeid"],
            "site_id": merged["cluster_id"],
            "source_site_id": merged["source_sample_siteid"],
            "sample_date": pd.to_datetime(merged["sample_date"], errors="coerce"),
            "sample_id": merged["sample_id"],
            "event_id": merged["eventid_epi"],
            "canonical_variable": merged["canonical_variable"],
            "canonical_label": merged["canonical_variable"].map(CANONICAL_LABELS),
            "lagos_parameter_id": merged["parameter_id"],
            "lagos_parameter_name": merged["parameter_name"],
            "standard_value": pd.to_numeric(merged["parameter_value"], errors="coerce"),
            "standard_unit_inferred": merged["canonical_variable"].map(STANDARD_UNITS),
            "raw_parameter_name": merged["source_parameter_name"],
            "raw_value": merged["source_parameter_value"],
            "raw_unit": merged["source_parameter_unit"],
            "censor_code": merged["censorcode"],
            "source_value_qualifier_code": merged["source_value_qualifiercode"],
            "standard_detection_limit_value": pd.to_numeric(
                merged["parameter_detectionlimit_value"], errors="coerce"
            ),
            "source_detection_limit_value": merged["source_detectionlimit_value"],
            "source_detection_limit_unit": merged["source_detectionlimit_unit"],
            "source_detection_limit_condition": merged["source_detectionlimit_condition"],
            "sample_depth_m": pd.to_numeric(merged["sample_depth_m"], errors="coerce"),
            "sample_depth_flag": merged["sample_depth_flag"],
            "site_lat_dd": pd.to_numeric(merged["cluster_lat_dd"], errors="coerce"),
            "site_lon_dd": pd.to_numeric(merged["cluster_lon_dd"], errors="coerce"),
            "source_id": merged["source_id"],
            "source_name": merged["source_name"],
            "source_activity_org_name": merged["source_activityorg_name"],
            "source_activity_id": merged["source_activityid"],
            "source_comments": merged["source_comments"],
            "parameter_conversion_factor": pd.to_numeric(
                merged["parameter_conversionfactor"], errors="coerce"
            ),
            "parameter_detection_limit_conversion_factor": pd.to_numeric(
                merged["parameter_detectionlimit_conversionfactor"], errors="coerce"
            ),
            "package_id": PACKAGE["package_id"],
            "package_doi": PACKAGE["doi"],
            "data_level": "raw_standardized_no_aggressive_cleaning",
        }
    )
    return output.sort_values(["lake_id", "site_id", "sample_date", "canonical_variable", "sample_id"])

def download_lagos(output, force=False, chunksize=500000):
    """Acquire EDI 1439.5 entities and export source-level lake measurements."""
    output=Path(output);raw=output/'lagos';raw.mkdir(parents=True,exist_ok=True)
    metadata=fetch_text(PACKAGE['metadata_url'])
    (raw/'metadata_eml.xml').write_text(metadata,encoding='utf8')
    entities=[e for e in parse_entities(metadata) if e.object_name in {CHEMISTRY_ENTITY,SOURCE_ENTITY}]
    if len(entities)!=2:raise ValueError('The two LAGOS source entities are missing')
    for entity in entities:download_file(entity,raw,force=force)
    chemistry,_=read_filtered_chemistry(raw/CHEMISTRY_ENTITY,chunksize)
    source=read_filtered_source(raw/SOURCE_ENTITY,set(chemistry.sample_id.dropna().astype(str)),chunksize)
    standardize_table(chemistry,source).to_parquet(output/'water_quality_raw.parquet',index=False)
