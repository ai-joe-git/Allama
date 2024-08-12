#!/bin/bash

# Function to print messages
print_message() {
    echo ">>> $1"
}

# Create a temporary directory
temp_dir=$(mktemp -d)
cd "$temp_dir"

# Download the installation script
print_message "Downloading Allama installation script..."
curl -s -O https://raw.githubusercontent.com/ai-joe-git/Allama/main/simple_llm.sh

# Make the script executable
print_message "Making the installation script executable..."
chmod +x simple_llm.sh

# Run the installation script
print_message "Running the Allama installation script..."
./simple_llm.sh

# Clean up
cd ..
rm -rf "$temp_dir"
