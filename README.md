# Proyecto para Automatización de Espectroscopía

## Descripción general

Este proyecto busca automatizar el flujo de trabajo de limpieza y análisis de espectros astronómicos, reduciendo el tiempo y esfuerzo manual que normalmente requieren estas tareas.

## Problemática

En espectroscopía, el proceso de limpieza de los espectros —que incluye la remoción de la contaminación telúrica y la normalización espectral— suele ser bastante extenso y engorroso cuando se realiza de forma manual. Esto ralentiza el análisis y aumenta el riesgo de errores humanos, especialmente cuando se trabaja con grandes volúmenes de datos.

## Herramientas

Para automatizar este proceso se utilizan las siguientes herramientas:

- **MOLECFIT**: herramienta utilizada para la extracción y corrección de las líneas telúricas presentes en los espectros.
- **Algoritmo de minimización de χ²**: una vez limpio el espectro, se aplica un algoritmo propio que automatiza el análisis espectroscópico mediante la minimización del estadístico de prueba $\chi^2$, permitiendo ajustar modelos sintéticos a los datos observados de forma eficiente.
- **Turbospectrum**: se recomienda su uso para la síntesis espectral, generando los espectros sintéticos que se comparan con las observaciones durante el proceso de minimización.

## Flujo de trabajo

1. **Entrada**: espectro crudo (sin procesar).
2. **Corrección telúrica**: MOLECFIT identifica y remueve las líneas de absorción telúrica.
3. **Normalización**: el espectro se normaliza para facilitar su análisis posterior.
4. **Síntesis espectral**: Turbospectrum genera los espectros sintéticos de referencia.
5. **Ajuste espectroscópico**: el algoritmo de minimización de χ² compara el espectro limpio con los modelos sintéticos generados.
6. **Salida**: espectro limpio y parámetro de metalicidad estimado.

## Resultados

Como resultado de este pipeline se obtiene:

- Un **espectro limpio**, libre de contaminación telúrica y normalizado, listo para su análisis espectroscópico.
- La **metalicidad estelar** expresada en [Fe/H] dex, obtenida a partir del ajuste automatizado.

## Uso

```bash
# Ejemplo de ejecución del pipeline (ajustar según implementación real)
python run_pipeline.py --input espectro_crudo.fits --output resultados/
```

## Estructura del proyecto

```
├── data/               # Espectros de entrada
├── molecfit_config/    # Archivos de configuración de MOLECFIT
├── turbospectrum/      # Configuración y modelos de síntesis espectral
├── scripts/            # Scripts de automatización
├── results/            # Espectros limpios y resultados de metalicidad
└── README.md
```

## Licencia

Especificar la licencia del proyecto aquí (por ejemplo, MIT, GPL-3.0, etc.).
