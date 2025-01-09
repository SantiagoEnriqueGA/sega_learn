# Find TODO comments in Python files, excluding paths containing "__archive"
$todoComments = Get-ChildItem -Recurse -Filter *.py | Where-Object { 
    -not ($_.FullName -like "*__archive*") 
} | ForEach-Object {
    $file = $_
    $lines = Get-Content $file.FullName
    $lines | ForEach-Object -Begin { $global:lineNumber = 0 } -Process {
        $lineNumber++
        $trimmedLine = $_.TrimStart()
        if ($trimmedLine -match 'TODO') {
            [PSCustomObject]@{
                FileName   = $file.FullName -replace ".*sega_learn", "sega_learn"
                LineNumber = $global:lineNumber
                Line       = $trimmedLine
            }
        }
    }
}

# Write TODO comments to the file
$todoComments | Tee-Object -FilePath "scripts/out/todo_comments.txt" | Format-Table -AutoSize

# Append the "Other" section to the file
# Add new TODOs here:
$otherTodos = @"
Other:
"@

Add-Content -Path "scripts/out/todo_comments.txt" -Value $otherTodos