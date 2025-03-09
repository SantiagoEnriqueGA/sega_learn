# Export the conda environment named 'sega_learn' to environment.yml
conda env export --no-builds -n sega_learn > environment.yml

# Remove the 'prefix' line from the environment.yml file
(Get-Content environment.yml) | Where-Object { $_ -notmatch '^prefix:' } | Set-Content environment.yml


# Export the conda environment named 'sega_learn_pypy' to environment_pypy.yml
conda env export --no-builds -n sega_learn_pypy > environment_pypy.yml

# Remove the 'prefix' line from the environment.yml file
(Get-Content environment_pypy.yml) | Where-Object { $_ -notmatch '^prefix:' } | Set-Content environment_pypy.yml
