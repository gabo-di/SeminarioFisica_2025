# Seminarios de Fisica 2025
Charla en FCyT 16-Sep-2025 a 20-Sep-2025

## Codigo Julia

Los codigos para reproducir los ejemplos de cada dia estan en las carpetas `day1`, `day2`, etc.

Inicializar julia con el proyecto
```
$ julia --project=./ 
```
dentro del REPL de julia:

```
julia> using Pkg
julia> Pkg.activate()
julia> Pkg.instantiate()
```
para correr el codigo recomendamos usar Revise para leer el codigo:

```
julia> using Revise
julia> includet("day1/main.jl")
julia> main_simple_harmonic_oscillator()
```

## Presentacion

El latex para generar la presentacion de cada dia esta en la carpeta `presentation`. 
Para realmente generar las presentaciones sin errores es necesario generar las figuras
con el codigo de cada dia.
