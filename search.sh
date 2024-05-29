#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 directory text"
    exit 1
fi

directory=$1
text=$2

# Check if the specified directory exists
if [ ! -d "$directory" ]; then
    echo "Directory $directory does not exist."
    exit 1
fi

# Function to count occurrences of the text in a file
count_occurrences() {
    local file=$1
    local text=$2
    local count=$(grep -o "$text" "$file" | wc -l)
    echo "Text '$text' appears $count times in $file"
}

export -f count_occurrences

# Use parallel to count occurrences in each file
find "$directory" -type f | parallel count_occurrences {} "$text"
