import os
import requests

# Example URL for data download: January 2011
# https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2011-01.parquet

# Local Data Hierarchy
# data/
# ├── 2010
# │   ├── yellow_tripdata_2010-01.parquet
# ....
# ├── 2011
# │   ├── yellow_tripdata_2011-01.parquet


DOWNLOAD_URL_TEMPLATE = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{{year}}-{{month:02d}}.parquet"
YEARS = [2011, 2012, 2013]


def main():
    for year in YEARS:
        os.makedirs(f"data/{year}", exist_ok=True)
        for month in range(1, 13):
            url = DOWNLOAD_URL_TEMPLATE.format(year=year, month=month)
            local_path = f"data/{year}/yellow_tripdata_{year}-{month:02d}.parquet"
            print(f"Downloading {url} to {local_path}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {local_path}")


if __name__ == "__main__":
    main()
