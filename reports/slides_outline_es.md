# Guion del deck (español)

11 diapositivas principales y 3 de apéndice, sobre `assets/template.pptx`. Los layouts se citan por índice (ver `reports/template_inspection.md`); los términos siguen `reports/glossary_es.md`. Cada número viene de un archivo en `reports/`, citado en "Fuentes". Las figuras están en `reports/figures/deck/` y se generan con `python -m intent.deck_figures`; cada una mide lo mismo que el área donde va, así que se insertan al 100%. Los dos diagramas (5 y 10) son formas nativas de PowerPoint.

`python -m intent.deck build` lee este archivo: título, layout, figuras (`*.png`), tablas markdown, `diagrama nativo` y texto de cada sección, y las notas del orador.

Uso de layouts:
- **8, Contenido + tabla:** texto a la izquierda (3.19 x 4.70) y figura o tabla a la derecha (8.30 x 4.70).
- **6, Dos contenidos 1:** dos columnas de 5.94 x 4.37. Se prefiere al layout 9 porque el 9 tiene una decoración dentro de su columna derecha.
- **10, Tabla:** un área completa de 12.13 x 4.70.
- **4, Título y contenido 1:** solo texto.

No se incluyen portada ni separador de apéndice (se pidieron 11 + 3); si se agregan, los layouts adecuados son 12 y 2.

---

## 1. Problema y contexto

- **Título:** El destino del crédito llega en texto libre; etiquetarlo automáticamente lo convierte en una señal de riesgo y de producto.
- **Layout:** 8 (Contenido + tabla)
- **Visual:** tabla nativa en el área derecha, con 4 respuestas reales y sus etiquetas. Son las mismas de la demo de la diapositiva 10 (filas de test, texto literal):

  | Respuesta | Etiquetas |
  |---|---|
  | Invertiré en equipo de transporte tractocamion | equ |
  | Compra de inventario publicidad y capacitacion | inv, mkt |
  | transferir otro crédito que está muy caro | cred |
  | Quiero irme de vacaciones en mayo | no |

- **Texto:**
  - 6,691 respuestas de solicitudes PyME, en español
  - 10 etiquetas; una respuesta puede tener varias
  - Métrica principal, fijada antes de modelar: macro F1
- **Notas del orador:** Konfio pregunta en la solicitud para qué se usará el crédito, y hoy esa respuesta casi no se aprovecha porque leer miles a mano no escala. Dos etiquetas son señales de riesgo, no (consumo personal) y cred (refinanciamiento): pasarlas por alto deja pasar un riesgo, mientras que una falsa alarma cuesta un minuto de revisión. Elegimos macro F1 para que etiquetas raras como temp, con 2.8% de las respuestas, cuenten igual que inv.
- **Fuentes:** report.md §1 y §2; 6,691 = 5,703 de CV + 988 de test (report.md, encabezado); app/examples.yaml.

## 2. Calidad de datos

- **Título:** Los datos traían defectos estructurales; los reparamos y documentamos en vez de descartar filas.
- **Layout:** 8 (Contenido + tabla)
- **Visual:** tabla nativa en el área derecha:

  | Problema | Alcance | Decisión |
  |---|---|---|
  | Registros fusionados en una celda por saltos de línea | unos 22 registros | recuperados línea por línea |
  | Mojibake (crÃ©dito en vez de crédito) | cerca de 42% de las filas | corregido con ftfy |
  | Texto sin sentido o no textual | 9 + 1 filas | eliminadas, con su motivo registrado |
  | Texto idéntico con etiquetas distintas | 58 grupos | se conservan, siempre del mismo lado de la partición |
  | inv no aparece en el codebook | 34.7% de las respuestas | supuesto: inventario o mercancía |

- **Texto:**
  - 6,679 filas en el archivo; el brief dice 6,726
  - 6,691 respuestas limpias tras reparar
  - 58 grupos contradictorios: ruido de anotación medible
- **Notas del orador:** El archivo trae 47 filas menos que el brief y una celda con unos 22 registros pegados por saltos de línea, rastro de una exportación CSV rota; los recuperamos línea por línea porque temp, la etiqueta más rara, no puede perder ejemplos. Cerca de 42% de las filas tenía la codificación rota y se corrigió con ftfy. Los 58 grupos de texto idéntico con etiquetas distintas son la primera evidencia de que parte del error estará en las etiquetas.
- **Fuentes:** report.md §2.

## 3. Estructura de etiquetas

- **Título:** Las etiquetas están muy desbalanceadas: inv aparece en 34.7% de las respuestas y temp solo en 2.8%.
- **Layout:** 8 (Contenido + tabla)
- **Visual:** figura `etiquetas_prevalencia.png` (8.3 x 4.7 in) en el área derecha.
- **Texto:**
  - 86% con una etiqueta, 12.6% con dos, 1% con tres
  - no casi nunca se combina: solo 4 respuestas
  - Respuestas cortas: mediana de 11 palabras
- **Notas del orador:** El desbalance explica dos decisiones: macro F1 como métrica y un umbral distinto por etiqueta, porque con 2.8% de prevalencia temp desaparecería en un promedio por fila. inv no viene en el codebook; asumimos que significa inventario o mercancía, y los textos etiquetados así lo confirman. Que las respuestas sean cortas, con p95 de 35 palabras, anticipa que unas pocas palabras clave deciden la etiqueta.
- **Fuentes:** report.md §2.

## 4. Diseño de evaluación

- **Título:** El test se separó al inicio y se usó una sola vez; modelo, mezcla y umbrales se eligieron con validación cruzada agrupada.
- **Layout:** 8 (Contenido + tabla)
- **Visual:** figura `evaluacion_particion.png` (8.3 x 4.7 in) en el área derecha.
- **Texto:**
  - Grupos por texto normalizado: sin duplicados entre lados
  - Estratificación iterativa: temp y sueldo en cada fold
  - Features de segunda etapa, siempre out-of-fold
- **Notas del orador:** Si un mismo texto cae en entrenamiento y en evaluación, el modelo se premia por memorizar; agrupar por texto normalizado lo impide. Lo medimos: con partición por filas, B2 sube de 0.671 a 0.679 de macro F1 en CV, una inflación pequeña pero real (apéndice A2). Todo lo que se ajustó, incluidos el peso de la mezcla y los umbrales, usa solo predicciones out-of-fold de los 5 folds.
- **Fuentes:** report.md §3; leakage_check.csv.

## 5. Enfoque

- **Título:** El modelo final mezcla un e5 con fine-tuning y TF-IDF con regresión logística, porque se equivocan en respuestas distintas.
- **Layout:** 10 (Tabla)
- **Visual:** diagrama nativo `pipeline` (formas y conectores de PowerPoint), ocupa el área de contenido.
- **Texto:** ninguno; el diagrama lleva sus propias etiquetas.
- **Notas del orador:** Cada familia de modelos recibe su propia rama de preprocesamiento: el transformer ve el texto casi original y TF-IDF ve lemas sin acentos ni stopwords. E9b promedia 50/50 las probabilidades de ambos y aplica un umbral por etiqueta, todo ajustado con predicciones out-of-fold. Las reglas finales garantizan al menos una etiqueta por respuesta y dejan no sola cuando es la más probable.
- **Fuentes:** report.md §3 y §4 (configuración congelada en configs/base.yaml, `selected_model`).

## 6. Ablación

- **Título:** TF-IDF es una línea base difícil de superar: solo el fine-tuning la supera, y mezclar ambos da el mejor resultado.
- **Layout:** 8 (Contenido + tabla)
- **Visual:** figura `ablacion_cv.png` (8.3 x 4.7 in) en el área derecha: 8 configuraciones con IC 95%.
- **Texto:**
  - Embeddings congelados: 0.601, debajo de B2 (0.671)
  - e5 con fine-tuning: 0.701; mezcla E9b: 0.715
  - Umbrales y reglas: de 0.664 a 0.671
- **Notas del orador:** Las respuestas son cortas y las deciden palabras como nómina, renta o mercancía, que los embeddings generales difuminan. Con fine-tuning el encoder aprende esas señales y se equivoca en respuestas distintas que TF-IDF: en CV, el e5 es mejor en 1,000 filas y B2 en 695, y por eso la mezcla gana. La tabla completa, con intervalos y los experimentos que no aparecen en la figura, está en el apéndice A1.
- **Fuentes:** ablation.md; report.md §4; results.csv.

## 7. Resultados en test frente a B2 y al baseline de la empresa

- **Título:** En test, E9b supera a B2 por 0.043 de macro F1 y su Hamming loss queda debajo del baseline de la empresa.
- **Layout:** 6 (Dos contenidos 1)
- **Visual:**
  - Izquierda: figura `test_f1.png` (5.9 x 3.6 in), con macro, micro y samples F1 e IC 95%.
  - Derecha: figura `test_hamming.png` (5.9 x 3.6 in), con E9b, B2 y la empresa.
  - Cada figura lleva una línea de texto debajo.
- **Texto:**
  - Diferencia pareada en macro F1: +0.043 [+0.020, +0.065]
  - Su "67% de precisión": mean PR-AUC 0.779, micro 0.749
- **Notas del orador:** El test de 988 filas se evaluó una sola vez y confirma el orden de CV: 0.732 en test contra 0.715 estimado en CV. La comparación con la empresa es indicativa, porque su test es otro y probablemente partió por filas, lo que según nuestro chequeo favorece un poco su cifra. Su "precisión promedio de 67%" es ambigua: mean PR-AUC y precisión micro la superan con claridad, y la precisión macro, 0.692 [0.660, 0.723], la supera sin ser concluyente.
- **Fuentes:** test_results.csv; test_paired_vs_B2.csv; report.md §5.

## 8. Vista por etiqueta y el problema de temp

- **Título:** temp es la etiqueta más débil, con F1 de 0.40 en test, porque su definición mezcla temporada y ciclo de cobro.
- **Layout:** 6 (Dos contenidos 1)
- **Visual:**
  - Izquierda: figura `test_f1_por_etiqueta.png` (5.9 x 4.3 in), F1 en test de E9b y B2 por etiqueta.
  - Derecha: figura `temp_lecturas.png` (5.9 x 3.6 in), qué mencionan las 161 respuestas temp de CV, con el texto debajo.
- **Texto:**
  - 17% de las respuestas temp pierde la etiqueta ante inv
  - 28 positivos en test: IC de F1 de 0.24 a 0.55
- **Notas del orador:** E9b mejora a B2 en las 10 etiquetas, con las mayores ganancias en no, crec y sueldo. En temp, muchas respuestas describen financiar el ciclo de cobro (contratos, facturas a 30 a 120 días) y 42% no menciona ni temporada ni ciclo, así que el techo lo pone la definición más que el modelo. Proponemos separar temp en temporada y ciclo de cobro en el codebook.
- **Fuentes:** test_per_label.csv; test_results.csv; report.md §2 y §6.

## 9. Análisis de errores y auditoría manual

- **Título:** La mayor parte del error restante está en las etiquetas y en el codebook, no en el modelo.
- **Layout:** 8 (Contenido + tabla)
- **Visual:** figura `auditoria_manual.png` (8.3 x 4.7 in) en el área derecha: causas de 50 errores revisados a mano, con IC 95% de Wilson.
- **Texto:**
  - Solo 12% de 50 errores revisados es del modelo
  - cleanlab marca 24.8% de CV: cota superior
  - equ vs inv: 102 equ predichos como inv, 53 al revés
- **Notas del orador:** La categoría más grande es de convención: la respuesta nombra varios usos y el modelo y el anotador eligen subconjuntos distintos. La revisión manual coincide con la clasificación automática en 80% de los casos (kappa 0.67), aunque la hizo una sola persona que veía la categoría automática, y por eso los intervalos son amplios. Un patrón pide una decisión de negocio: los anotadores infieren inv por el giro, como en "Invertir en mi tienda de abarrotes"; B2 lo imita y el e5 lo lee literalmente como crecimiento.
- **Fuentes:** manual_audit.csv; report.md §6.

## 10. Despliegue y demo

- **Título:** E9b responde en 31.7 ms en CPU y manda a revisión humana los casos de riesgo o cercanos al umbral.
- **Layout:** 6 (Dos contenidos 1)
- **Visual:**
  - Izquierda: diagrama nativo `despliegue`. Muestra POST /predict; E9b en CPU con p50 31.7 ms, p95 47.6 ms y B2 de respaldo en 5.3 ms; la respuesta con etiquetas, probabilidades y needs_review; la revisión humana (a 0.1 del umbral, o no / cred) o la etiqueta automática; y el monitoreo y el reentrenamiento.
  - Derecha: captura de la demo, `demo_app.png` (ejemplo "Riesgo cred", tomado con `python app/screenshot.py`), con la marca "Demo en vivo: misma API, sin conexión a internet".
- **Texto:** ninguno aparte de las etiquetas del diagrama y la marca de la demo.
- **Notas del orador:** Demo en vivo: abrir la demo con `make demo`, mostrar el ejemplo "Riesgo cred" y luego escribir una respuesta nueva. needs_review se activa cuando alguna probabilidad queda a 0.1 de su umbral o cuando se predice no o cred, porque ambas cambian la decisión de crédito. Los umbrales actuales maximizan F1 y son un punto de partida que el negocio debe fijar según su costo de error, y B2 queda de respaldo cuando importan la latencia o el costo de GPU, a cambio de 0.043 de macro F1 en test.
- **Fuentes:** latency.csv; report.md §5 y §7; configs/base.yaml (`serving`).

## 11. Siguientes pasos

- **Título:** La siguiente mejora más barata está en el codebook y en las etiquetas, no en un modelo más grande.
- **Layout:** 4 (Título y contenido 1)
- **Visual:** ninguno; las tres líneas van en el área de contenido.
- **Texto:**
  - 1. Codebook: definir inv, separar temp, crec como residual
  - 2. Reetiquetar lo que marca cleanlab, con aprendizaje activo
  - 3. Umbrales por costo para no y cred, con monitoreo
- **Notas del orador:** Definir inv como bienes para reventa e insumos, con los activos durables en equ, ataca la confusión más grande del modelo. Después conviene reetiquetar las filas marcadas por cleanlab, empezando por las de menor calidad, y usar un LLM solo como segundo anotador que señale desacuerdos, nunca como verdad. En producción, el negocio fija la recall objetivo de no y cred y acepta la carga de revisión que implica.
- **Fuentes:** report.md §7 y §8.

---

## A1. Tabla completa de ablación con intervalos

- **Título:** Apéndice A1. Ablación completa: el intervalo de E9b no se traslapa con el de B2.
- **Layout:** 10 (Tabla)
- **Visual:**
  - Tabla nativa de 23 filas más encabezado, en letra de unos 10 pt.
  - El bloque va como primera columna con celdas combinadas, en lugar de filas de encabezado.
  - E9b va en negritas.

  | Bloque | Experimento | Qué cambia | Macro F1 [IC 95%] | Micro F1 | Hamming | temp F1 |
  |---|---|---|---|---|---|---|
  | Líneas base | B0 | No predecir ninguna etiqueta | 0.000 [0.000, 0.000] | 0.000 | 0.1143 | 0.000 |
  | Líneas base | B1 | Siempre la etiqueta más frecuente (inv) | 0.051 [0.050, 0.053] | 0.323 | 0.1451 | 0.000 |
  | TF-IDF + LR | B2, umbral 0.5 | Sin umbrales ajustados ni reglas | 0.664 [0.652, 0.676] | 0.726 | 0.0683 | 0.370 |
  | TF-IDF + LR | B2, umbrales | + umbrales ajustados out-of-fold | 0.666 [0.654, 0.678] | 0.735 | 0.0639 | 0.374 |
  | TF-IDF + LR | B2, al menos una | + regla de al menos una etiqueta | 0.668 [0.656, 0.680] | 0.734 | 0.0658 | 0.378 |
  | TF-IDF + LR | B2, no excluyente | + regla de no excluyente (sin la anterior) | 0.669 [0.657, 0.682] | 0.738 | 0.0630 | 0.380 |
  | TF-IDF + LR | B2 (referencia) | Umbrales + ambas reglas | 0.671 [0.659, 0.684] | 0.736 | 0.0650 | 0.384 |
  | TF-IDF + LR | B2_margin | Al menos una, por mayor margen | 0.673 [0.661, 0.686] | 0.738 | 0.0646 | 0.401 |
  | TF-IDF + LR | B3 | Cadena de clasificadores | 0.665 [0.652, 0.678] | 0.735 | 0.0642 | 0.372 |
  | TF-IDF + LR | B2_codebook | B2 + 10 similitudes con el codebook | 0.666 [0.654, 0.679] | 0.731 | 0.0655 | 0.392 |
  | Embeddings congelados | E1_e5 | e5 + LR, umbral 0.5 | 0.580 [0.568, 0.592] | 0.653 | 0.0888 | 0.236 |
  | Embeddings congelados | E2_e5 | + similitudes con el codebook | 0.583 [0.570, 0.594] | 0.655 | 0.0881 | 0.238 |
  | Embeddings congelados | E3_e5 | + umbrales y reglas | 0.601 [0.589, 0.613] | 0.672 | 0.0811 | 0.250 |
  | Embeddings congelados | E1_mpnet | mpnet + LR, umbral 0.5 | 0.574 [0.561, 0.586] | 0.640 | 0.0933 | 0.288 |
  | Embeddings congelados | E2_mpnet | + similitudes con el codebook | 0.573 [0.560, 0.585] | 0.641 | 0.0931 | 0.294 |
  | Embeddings congelados | E3_mpnet | + umbrales y reglas | 0.588 [0.574, 0.601] | 0.651 | 0.0871 | 0.291 |
  | e5 con fine-tuning | E4 | Probabilidades de la cabeza | 0.701 [0.688, 0.714] | 0.758 | 0.0575 | 0.410 |
  | e5 con fine-tuning | E5 | Embedding con fine-tuning + LR | 0.637 [0.625, 0.649] | 0.693 | 0.0749 | 0.320 |
  | e5 con fine-tuning | E6 | Embedding + probabilidades B2 + codebook, LR | 0.642 [0.629, 0.653] | 0.693 | 0.0762 | 0.271 |
  | e5 con fine-tuning | E7 | Mismas features, LightGBM | 0.702 [0.691, 0.715] | 0.759 | 0.0578 | 0.355 |
  | Estructura y mezcla | E8 | no contra negocio primero, luego E7 | 0.693 [0.680, 0.706] | 0.753 | 0.0585 | 0.377 |
  | Estructura y mezcla | E9 | Mezcla B2 + E7 | 0.712 [0.700, 0.724] | 0.766 | 0.0562 | 0.399 |
  | Estructura y mezcla | **E9b (seleccionado)** | **Mezcla B2 + E4** | **0.715 [0.703, 0.727]** | **0.771** | **0.0547** | **0.421** |

- **Texto:** ninguno.
- **Notas del orador:** Todos los números son predicciones out-of-fold de CV con intervalos por bootstrap de grupos; el test no participa. E9b [0.703, 0.727] y B2 [0.659, 0.684] no se traslapan, y E9 (0.712) queda cerca de E9b (0.715); E9b además no entrena nada sobre features de etapa 1, así que no le afecta la fuga indirecta que sí tienen los modelos de dos etapas. Los diagnósticos no comparables, como la partición por filas, están en el apéndice A2.
- **Fuentes:** ablation.md; report.md §4.

## A2. Chequeo de fuga y trazabilidad

- **Título:** Apéndice A2. Partir por filas infla poco las métricas, y la selección quedó fija antes del único uso del test.
- **Layout:** 6 (Dos contenidos 1)
- **Visual:**
  - Izquierda: tabla nativa con B2 en CV:

    | Métrica | Partición agrupada | Partición por filas |
    |---|---|---|
    | Macro F1 | 0.671 | 0.679 |
    | Hamming loss | 0.0650 | 0.0633 |
    | Samples F1 | 0.748 | 0.751 |
    | temp F1 | 0.384 | 0.420 |
    | Samples F1 en las 199 filas con duplicado en otro fold | 0.701 | 0.714 |

  - Derecha: figura `trazabilidad_linea_tiempo.png` (5.9 x 4.3 in).
- **Texto:** ninguno; el dato de las 199 filas va en la tabla.
- **Notas del orador:** Con partición por filas, 199 filas de CV tienen un duplicado en otro fold, y temp es la que más se infla, de 0.384 a 0.420. El hash 1f3cf3b6c783 lo escribió la propia corrida de test, y solo lo reproduce una config que ya contenía la selección de E9b. El commit y el tag son posteriores al test: registran el estado, no prueban el orden.
- **Fuentes:** leakage_check.csv; TRACEABILITY.md; test_results.csv (`config_sha`).

## A3. Métricas por etiqueta en test

- **Título:** Apéndice A3. En test, E9b supera a B2 en F1 en las 10 etiquetas; temp sigue siendo la más baja.
- **Layout:** 10 (Tabla)
- **Visual:** tabla nativa, 10 filas:

  | Etiqueta | Descripción | Precisión | Recall | F1 | PR-AUC | Soporte | F1 de B2 |
  |---|---|---|---|---|---|---|---|
  | crec | crecimiento sin uso específico | 0.587 | 0.693 | 0.635 | 0.697 | 88 | 0.553 |
  | cred | pago de deudas | 0.779 | 0.828 | 0.803 | 0.856 | 64 | 0.791 |
  | equ | equipo | 0.837 | 0.843 | 0.840 | 0.912 | 299 | 0.831 |
  | inic | iniciar un negocio | 0.651 | 0.719 | 0.683 | 0.728 | 57 | 0.630 |
  | inv | inventario o mercancía | 0.797 | 0.903 | 0.847 | 0.911 | 349 | 0.836 |
  | mkt | marketing | 0.842 | 0.941 | 0.889 | 0.967 | 51 | 0.873 |
  | no | no destinado a capital de trabajo | 0.839 | 0.788 | 0.812 | 0.836 | 66 | 0.719 |
  | renta | renta | 0.638 | 0.779 | 0.701 | 0.739 | 113 | 0.677 |
  | sueldo | nómina | 0.580 | 0.906 | 0.707 | 0.757 | 32 | 0.639 |
  | temp | ventas de temporada | 0.375 | 0.429 | 0.400 | 0.393 | 28 | 0.338 |

- **Texto:** ninguno.
- **Notas del orador:** sueldo tiene precisión 0.58 con recall 0.91: si una falsa alarma de nómina es costosa, su umbral debe subir. crec, renta y temp tienen precisión por debajo de 0.65, consistente con definiciones difusas. mkt, equ e inv son las más estables, con PR-AUC arriba de 0.9.
- **Fuentes:** test_per_label.csv; report.md §5.

---

## Figuras del deck

| Figura | Diapositiva | Tamaño (in) | Datos |
|---|---|---|---|
| `etiquetas_prevalencia.png` | 3 | 8.3 x 4.7 | data/interim/clean.parquet; verificada contra report.md §2 |
| `evaluacion_particion.png` | 4 | 8.3 x 4.7 | data/processed/folds.parquet; 5,703 y 988 verificados |
| `ablacion_cv.png` | 6 | 8.3 x 4.7 | results.csv (filas de CV) |
| `test_f1.png` | 7 | 5.9 x 3.6 | test_results.csv |
| `test_hamming.png` | 7 | 5.9 x 3.6 | test_results.csv |
| `test_f1_por_etiqueta.png` | 8 | 5.9 x 4.3 | test_per_label.csv |
| `temp_lecturas.png` | 8 | 5.9 x 3.6 | `evaluate.temp_readings` sobre CV; verificada contra report.md §2 |
| `auditoria_manual.png` | 9 | 8.3 x 4.7 | manual_audit.csv, intervalos de Wilson |
| `demo_app.png` | 10 | captura | demo en vivo, ejemplo "Riesgo cred"; `python app/screenshot.py` |
| `trazabilidad_linea_tiempo.png` | A2 | 5.9 x 4.3 | TRACEABILITY.md; results.csv; test_results.csv |

Cada función de `src/intent/deck_figures.py` compara sus números con los del reporte y se niega a dibujar si no coinciden. Las figuras en inglés de `reports/figures/` no cambian.
