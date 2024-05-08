#!/bin/bash

# Base URL with placeholder for the number
base_url="https://www.gutenberg.org/cache/epub/%d/pg%d-images.html"

# Loop from 0 to 73563 and print formatted URLs
for num in $(seq 1 73563); do
    printf "$base_url\n" $num $num
done