# Bitcoin Halving Cycles — Modelo de Proyección

## Conclusiones del diseño

### El modelo central

Cada ciclo `n` es una instancia de la misma curva universal, escalada por dos funciones progresivas:

```
F(d, n) = forma(d / L(n)) * M(n)
```

- `d` — día dentro del ciclo (desde el halving)
- `n` — número de ciclo (1=2012, 2=2016, 3=2020, 4=2024)
- `L(n)` — largo del ciclo en días, **crece** con n
- `M(n)` — magnitud (múltiplo indexado), **decae** con n
- `forma(·)` — curva universal normalizada, **igual para todos los ciclos**

### Supuestos acordados

1. Los límites del eje X están fijos — los halvings son deterministas, no se modelan
2. Los ciclos se alargan horizontalmente con cada n
3. Los ciclos decaen en magnitud con cada n
4. La forma normalizada es universal — si L(n) y M(n) se modelan correctamente, las curvas colapsan en una sola forma
5. Los datos del ciclo 2012 son tan confiables como los demás — no se penalizan

### Estructura de censura

- **Ciclo 2012 (n=1)**: truncado a la izquierda — no hay datos de Bitcoin antes de que tuviera precio registrable
- **Ciclo 2024 (n=4)**: truncado a la derecha — ciclo en curso, datos hasta d+797
- **Ciclos 2016 y 2020 (n=2, n=3)**: completos — anclan el modelo

La censura se resuelve como consecuencia del modelo, no como problema separado.

### Flujo de implementación

1. Estimar `L(n)` y `M(n)` de los 4 ciclos
2. Normalizar cada ciclo → colapsan en `forma(·)`
3. Ajustar `forma(·)` sobre las curvas colapsadas
4. Proyectar ciclo 2024: evaluar `forma(d / L(4)) * M(4)` para `d > 797`

## Estimación (modelo_avanzado.py)

- **φ**: se promedia z = Y/A por bins de t (80 bins, primero por ciclo, después entre ciclos) y se ajusta spline cúbica con λ por GCV (`make_smoothing_spline`). No ajustar spline sobre los puntos crudos concatenados: los t casi-duplicados entre ciclos la rompen (φ plana con picos espurios — bug histórico ya corregido).
- **A(n)**: mínimos cuadrados alternados con φ — regresión sin intercepto `A = Σφ·Y/Σφ²`. Nunca usar `max(Y)`: sesgado hacia arriba en ciclos completos y hacia abajo en el ciclo censurado. Para el ciclo 2024, A(4) sale de la parte observada.
- **Incertidumbre**: bootstrap circular por bloques de residuos (8 semanas), refit ALS completo por iteración. Sin `except: pass` — si el fit falla debe crashear, no producir NaN silenciosos.
- **Decaimiento A(n)·L(n)**: pow y exp se reportan como escenarios separados, nunca promediados — con 3 puntos son indistinguibles in-sample.
- **Validación**: train ciclos 1+2 → predice ciclo 3, siempre comparando contra baselines (recta, ciclo anterior reescalado).

## Archivos

- `base.py` — proyección simple por promedio (línea base)
- `modelo_avanzado.py` — modelo completo Y=A·φ(d/L): estimación, validación, proyección y gráficos
