import re
import csv

# Open the text file and read the contents
with open('biosample_result.txt', 'r') as file:
    contents = file.read()

# Split the contents into individual records
records = contents.strip().split('\n\n')

# Define the regular expressions to extract the information
record_pattern = re.compile(r'^(\w+)\.\s+(.*?)\s+Identifiers:\s*BioSample:\s*(\w+);\s*SRA:\s*(\w+);\s*GEO:\s*(\w+)\s+Organism:\s*(\w+(?:\s+\w+)*)\s+Accession:\s*(\w+)\s+ID:\s*(\w+)$')

# Extract the information from each record and write it to a CSV file
with open('output_file.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Sample Name', 'Identifier', 'Organism', 'Accession', 'ID'])
    for record in records:
        match = record_pattern.match(record.strip())
        if match:
            sample_name = match.group(2)
            identifier = match.group(3)
            organism = match.group(6)
            accession = match.group(7)
            id = match.group(8)
            writer.writerow([sample_name, identifier, organism, accession, id])
