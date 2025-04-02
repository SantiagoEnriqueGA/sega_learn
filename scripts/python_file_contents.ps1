# Define the folders to search
$folders = @(
    ".\examples",
    ".\sega_learn",
    ".\tests"
)

# Define the output file
$outputFile = "scripts\out\python_file_contents.txt"

# Clear the output file if it exists
if (Test-Path $outputFile) {
    Remove-Item $outputFile
}

# Iterate through each folder
foreach ($folder in $folders) {
    # Get all Python files in the folder
    $files = Get-ChildItem -Path $folder -Recurse -Filter "*.py"

    foreach ($file in $files) {
        # Write the file path and content to the output file
        "`n-----------------------------------------------------------------" | Out-File -Append -FilePath $outputFile
        "$($file.FullName -replace '.*sega_learn', 'sega_learn'):" | Out-File -Append -FilePath $outputFile
        "-----------------------------------------------------------------" | Out-File -Append -FilePath $outputFile
        Get-Content -Path $file.FullName | Out-File -Append -FilePath $outputFile
    }
}
