# Find TODO comments in Python files, excluding paths containing "__archive"
Get-ChildItem -Recurse -Filter *.py | Where-Object { 
    -not ($_.FullName -like "*__archive*") 
} | ForEach-Object {
    $file = $_
    $lines = Get-Content $file.FullName
    $lines | ForEach-Object -Begin { $global:lineNumber = 0 } -Process {
        $lineNumber++
        $trimmedLine = $_.TrimStart()
        if ($trimmedLine -match 'TODO') {
            $todoComment = [PSCustomObject]@{
                FileName   = $file.FullName -replace ".*sega_learn", "sega_learn"
                LineNumber = $global:lineNumber
                Line       = $trimmedLine
            }
            $todoComment
        }
    }
} | Tee-Object -FilePath "scripts/out/todo_comments.txt" | Format-Table -AutoSize