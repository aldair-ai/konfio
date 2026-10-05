# Glosario inglés a español para el deck

Una sola traducción por término, usada igual en títulos, texto, notas del orador, tablas y figuras (`src/intent/deck_figures.py` usa las mismas descripciones de etiquetas). Los términos técnicos que en México se usan en inglés se dejan en inglés.

## Convenciones

- Punto decimal y coma de miles: 0.715, 6,691. Porcentajes con punto: 2.8%.
- Sin guiones largos. Rangos con "a": "30 a 120 días", "15:11 a 15:12".
- Los códigos de etiqueta van tal cual, en minúsculas (inv, equ, temp). La primera vez que aparece uno en una diapositiva puede ir con su descripción: "temp (ventas de temporada)".
- Los nombres de experimentos no se traducen: B2, B3, E3, E4, E7, E8, E9b.
- Intervalos: "IC 95%", escritos como [0.702, 0.759].

## Etiquetas

| Código | Report (inglés) | Deck (español) |
|---|---|---|
| crec | growth without a specific plan | crecimiento sin uso específico |
| cred | pay debts | pago de deudas |
| equ | equipment | equipo |
| inic | start a business | iniciar un negocio |
| inv | inventory or merchandise (assumed) | inventario o mercancía; se aclara una vez que es un supuesto (el codebook no la define) |
| mkt | marketing | marketing |
| no | use not destined to working capital | no destinado a capital de trabajo |
| renta | rent | renta |
| sueldo | payroll | nómina |
| temp | seasonal sales | ventas de temporada |

## Negocio y datos

| Inglés | Español | Nota |
|---|---|---|
| SME | PyME | |
| use of proceeds | destino del crédito | |
| free-text answer | respuesta en texto libre | |
| credit application | solicitud de crédito | |
| underwriting | originación | |
| working capital | capital de trabajo | |
| personal consumption | consumo personal | |
| refinancing | refinanciamiento | |
| receivables cycle, payment cycle | ciclo de cobro | facturas a 30 a 120 días, contratos, pedidos |
| season | temporada | |
| risk signal | señal de riesgo | no y cred |
| brief | brief | el documento del ejercicio |
| codebook | codebook | |
| label | etiqueta | |
| multi-label classification | clasificación multietiqueta | |
| label cardinality | etiquetas por respuesta | |
| prevalence | prevalencia | en texto corrido: "aparece en 34.7% de las respuestas" |
| co-occurrence | coocurrencia | |
| mutually exclusive | exclusiva | "no es casi exclusiva" |
| row | fila | |
| record | registro | |
| merged cell | celda con registros fusionados | |
| mojibake | mojibake | se explica una vez: "texto con codificación rota" |
| gibberish | texto sin sentido | |
| duplicate | duplicado | |
| conflicting duplicates | duplicados contradictorios | texto idéntico con etiquetas distintas |
| annotation noise | ruido de anotación | |
| annotator | anotador | |
| label error | error de etiqueta | |
| labeling convention | convención de etiquetado | |

## Evaluación

| Inglés | Español | Nota |
|---|---|---|
| train/test split | partición | |
| group-aware split, grouped split | partición agrupada | |
| row-level split | partición por filas | |
| group key, normalized text | grupo, texto normalizado | ftfy, minúsculas, espacios colapsados |
| held-out test set | test | "el test se usó una sola vez" |
| cross-validation (CV) | validación cruzada (CV) | |
| fold | fold | |
| out-of-fold (OOF) | out-of-fold | |
| iterative stratification | estratificación iterativa | |
| leakage | fuga | "chequeo de fuga" |
| primary metric | métrica principal | |
| macro F1, micro F1, samples F1 | macro F1, micro F1, samples F1 | sin traducir |
| Hamming loss | Hamming loss | |
| PR-AUC, mean PR-AUC | PR-AUC, mean PR-AUC | |
| precision | precisión | |
| recall | recall | |
| support | soporte | número de positivos |
| exact match ratio | coincidencia exacta | |
| confidence interval | intervalo de confianza (IC 95%) | |
| group bootstrap | bootstrap por grupos | |
| paired difference, paired bootstrap | diferencia pareada, bootstrap pareado | |
| baseline | línea base | |
| company baseline | baseline de la empresa | nombre propio de la cifra 0.06821 |
| reference (B2) | referencia | |
| all-zeros baseline | no predecir ninguna etiqueta | |
| upper bound | cota superior | |
| traceability | trazabilidad | |
| config hash | hash de la config | |

## Modelado

| Inglés | Español | Nota |
|---|---|---|
| ablation | ablación | |
| experiment | experimento | |
| selected model | modelo seleccionado | |
| preprocessing branch | rama de preprocesamiento | "rama léxica", "rama transformer" |
| lemmatization | lematización | |
| stopwords | stopwords | |
| TF-IDF, word / character n-grams | TF-IDF, n-gramas de palabras / caracteres | |
| logistic regression (LR) | regresión logística (LR) | |
| binary relevance | una regresión por etiqueta | |
| classifier chain | cadena de clasificadores | |
| frozen embeddings | embeddings congelados | |
| sentence embeddings | embeddings de oraciones | |
| fine-tuning, fine-tuned e5 | fine-tuning, e5 con fine-tuning | |
| encoder | encoder | |
| stage 1 / stage 2 | etapa 1 / etapa 2 | |
| classification head, head probabilities | cabeza de clasificación, probabilidades de la cabeza | |
| hybrid features | features híbridas | |
| hierarchical model | modelo jerárquico | |
| blend, blend weight | mezcla, peso de la mezcla | |
| threshold, per-label threshold | umbral, umbral por etiqueta | |
| decoding rules | reglas de decodificación | |
| at-least-one rule | regla de al menos una etiqueta | |
| no-exclusive rule | regla de no excluyente | no anula las demás cuando es la más probable |
| fallback | respaldo | B2 como modelo de respaldo |
| label quality (cleanlab) | calidad de etiqueta | |
| flagged rows | filas marcadas | |
| confusion (equ vs inv) | confusión | "102 equ predichos como inv" |

## Despliegue

| Inglés | Español | Nota |
|---|---|---|
| deployment, serving | despliegue | |
| endpoint, API | endpoint, API | POST /predict sin traducir |
| needs_review | needs_review | nombre del campo |
| human review, review queue | revisión humana, cola de revisión | |
| latency, p50, p95 | latencia, mediana, p95 | |
| monitoring | monitoreo | |
| label mix | mezcla de etiquetas | |
| confidence drift | deriva de confianza | |
| retraining | reentrenamiento | |
| active learning | aprendizaje activo | |
| relabeling | reetiquetado | |
| LLM-assisted labeling audit | LLM como segundo anotador | |
| business-owned threshold policy | umbrales definidos por el negocio | |
| demo | demo | |
