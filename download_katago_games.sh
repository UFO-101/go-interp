#!/bin/bash

download_dir="/weka/linear-probing/katago-training-sgfs/2022-11-01-to-2022-11-30-raw"
decompress_dir="/weka/linear-probing/katago-training-sgfs/2022-11-01-to-2022-11-30"
start_date="2021-12-01"
end_date="2021-12-31"

current_date=$start_date

while [[ "$current_date" < "$end_date" ]] || [[ "$current_date" == "$end_date" ]]
do
    file_name="${current_date}sgfs.tar.bz2"
    url="https://katagoarchive.org/kata1/traininggames/$file_name"
    
    # Download the file to the download directory
    wget -P "$download_dir" "$url"

    # Decompress the file to the decompress directory
    tar -xjf "${download_dir}/$file_name" -C "$decompress_dir"

    # Increment the date by 1 day
    current_date=$(date -I -d "$current_date + 1 day")
done