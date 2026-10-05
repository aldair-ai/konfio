# Template inspection

Source: `assets/template.pptx`, read with python-pptx by `python -m intent.deck inspect`. The template's own 13 sample slides (text and photos) are unrelated to this project: the deck uses only its layouts, theme colors and fonts.

## Slide and master

- Slide size: 13.33 x 7.50 in (16:9).
- Background: the master fills every layout with `bg2`, which the color map sends to `dk2` = **#1B192E** (dark navy). No layout overrides it.
- Default text color: `tx1` maps to `lt1` = **#FFFFFF**. The theme is dark: figures must use light ink on the navy, not dark ink on white.
- Default sizes: title 48 pt, body level 1 24 pt, body level 2 16 pt.

## Theme colors

Scheme name: Float.

| Slot | Hex | Role in the template |
|---|---|---|
| dk1 | `#000000` | unused by the master (bg1 maps here) |
| lt1 | `#FFFFFF` | all text (tx1) |
| dk2 | `#1B192E` | slide background (bg2) |
| lt2 | `#EAE5EB` | secondary text (tx2) |
| accent1 | `#13BE89` | green |
| accent2 | `#12B1BF` | teal; table header fill in the sample slides |
| accent3 | `#D40AA8` | magenta |
| accent4 | `#B86E62` | rose |
| accent5 | `#A3A3C1` | lavender gray |
| accent6 | `#37335B` | deep plum |
| hlink | `#0066FF` | hyperlinks |
| folHlink | `#666699` | visited hyperlinks |

## Fonts

| Role | Theme font | Installed here | Used in figures |
|---|---|---|---|
| Headings (major) | Walbaum Display | no (Office cloud font) | none: figures carry no titles; closest installed Didone is Bodoni MT |
| Body (minor) | Gill Sans MT | yes | every label |

Gill Sans MT draws the digit 1 as a bare stroke, so `11.5%` can read as `II.5%`. Figures keep it for consistency with the slide text; the config hash on the traceability figure is set in Consolas.

## Layouts

Positions and sizes in inches (left, top; width x height). Every layout except 0 and 7 also has date (idx 10), footer (idx 11) and slide number (idx 12) placeholders at the bottom edge.

| Index | Name | Content placeholders | Decorations (keep clear) |
|---|---|---|---|
| 0 | Solo el título | title idx 0 (5.06 x 6.06); picture idx 13 (6.85 x 7.50) | none |
| 1 | Agenda | title idx 0 (12.13 x 1.84); object idx 13 (12.13 x 4.28) | Grupo 22 at (0.67, 5.83), 0.74 x 1.08<br>Grupo 4 at (9.85, 1.66), 0.80 x 0.83<br>Elipse 10 at (11.12, 0.42), 1.18 x 1.18<br>Grupo 11 at (9.91, 3.50), 3.93 x 2.86<br>Grupo 16 at (6.22, 4.48), 1.46 x 1.38 |
| 2 | Título de sección | center_title idx 0 (12.16 x 2.50); picture idx 13 (13.33 x 4.12) | Elipse 6 at (1.34, 4.21), 0.39 x 0.39 |
| 3 | Título + subtítulo + imagen | center_title idx 0 (4.55 x 3.45); subtitle idx 1 (4.55 x 2.57); picture idx 13 (5.61 x 5.61) | Elipse 13 at (6.20, 5.53), 1.18 x 1.18<br>Grupo 9 at (11.84, 4.53), 0.80 x 0.83 |
| 4 | Título y contenido 1 | title idx 0 (8.71 x 1.59); object idx 1 (8.67 x 4.35) | Grupo 6 at (9.91, 0.82), 3.93 x 2.86<br>Grupo 18 at (9.54, 5.57), 1.46 x 1.38<br>Grupo 11 at (0.67, 5.83), 0.74 x 1.08 |
| 5 | Título + subtítulo | picture idx 13 (13.33 x 7.50); center_title idx 0 (10.00 x 2.50); subtitle idx 1 (10.00 x 2.50) | none |
| 6 | Dos contenidos 1 | title idx 0 (12.13 x 1.46); object idx 1 (5.94 x 4.37); object idx 13 (5.94 x 4.37) | Elipse 21 at (12.11, 0.36), 0.39 x 0.39<br>Grupo 12 at (0.36, 6.05), 0.69 x 0.73 |
| 7 | Contenido + imagen | title idx 0 (5.36 x 2.55); object idx 1 (5.38 x 3.76); picture idx 13 (6.67 x 7.50) | Grupo 3 at (5.25, 0.75), 0.80 x 0.83<br>Elipse 14 at (1.30, 5.72), 1.18 x 1.18 |
| 8 | Contenido + tabla | title idx 0 (12.13 x 1.35); object idx 1 (3.19 x 4.70); table idx 13 (8.30 x 4.70) | none |
| 9 | Dos contenidos | title idx 0 (12.13 x 1.46); object idx 1 (5.94 x 4.37); object idx 13 (5.84 x 4.37) | Elipse 21 at (12.11, 0.36), 0.39 x 0.39<br>Grupo 10 at (11.35, 2.22), 1.46 x 1.38<br>Forma libre: Forma 21 at (4.70, 0.00), 0.39 x 0.30<br>Grupo 12 at (0.36, 6.05), 0.69 x 0.73 |
| 10 | Tabla | title idx 0 (12.13 x 1.30); object idx 1 (12.13 x 4.70) | none |
| 11 | Cierre | title idx 0 (5.66 x 3.11); object idx 1 (5.66 x 2.96); picture idx 13 (6.18 x 6.33) | Elipse 6 at (5.80, 5.94), 1.18 x 1.18<br>Grupo 8 at (0.65, 6.14), 0.73 x 0.69<br>Grupo 11 at (0.67, 5.83), 0.74 x 1.08 |
| 12 | Diapositiva de título | center_title idx 0 (9.06 x 3.23); subtitle idx 1 (9.06 x 2.80) | Forma libre: Forma 18 at (0.67, 0.53), 1.18 x 1.38<br>Elipse 19 at (0.69, 0.91), 0.59 x 1.18<br>Elipse 24 at (1.97, 2.70), 0.39 x 0.39<br>Grupo 33 at (1.45, 4.95), 2.17 x 1.49 |

### Placeholder geometry

**0. Solo el título**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | TITLE | 7.67 | 0.72 | 5.06 | 6.06 |
| 13 | PICTURE | 0.00 | 0.00 | 6.85 | 7.50 |

**1. Agenda**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | TITLE | 0.60 | 0.53 | 12.13 | 1.84 |
| 13 | OBJECT | 0.60 | 2.65 | 12.13 | 4.28 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |

**2. Título de sección**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | CENTER_TITLE | 0.60 | 4.42 | 12.16 | 2.50 |
| 13 | PICTURE | 0.00 | 0.01 | 13.33 | 4.12 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |

**3. Título + subtítulo + imagen**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | CENTER_TITLE | 0.60 | 0.22 | 4.55 | 3.45 |
| 1 | SUBTITLE | 0.60 | 3.83 | 4.55 | 2.57 |
| 13 | PICTURE | 6.21 | 0.86 | 5.61 | 5.61 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |

**4. Título y contenido 1**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | TITLE | 0.60 | 0.55 | 8.71 | 1.59 |
| 1 | OBJECT | 0.64 | 2.30 | 8.67 | 4.35 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |

**5. Título + subtítulo**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 13 | PICTURE | 0.00 | 0.00 | 13.33 | 7.50 |
| 0 | CENTER_TITLE | 1.67 | 1.51 | 10.00 | 2.50 |
| 1 | SUBTITLE | 1.67 | 4.16 | 10.00 | 2.50 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |

**6. Dos contenidos 1**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | TITLE | 0.60 | 0.56 | 12.13 | 1.46 |
| 1 | OBJECT | 0.60 | 2.29 | 5.94 | 4.37 |
| 13 | OBJECT | 6.79 | 2.29 | 5.94 | 4.37 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |

**7. Contenido + imagen**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | TITLE | 0.66 | 0.22 | 5.36 | 2.55 |
| 1 | OBJECT | 0.64 | 3.09 | 5.38 | 3.76 |
| 13 | PICTURE | 6.67 | 0.00 | 6.67 | 7.50 |

**8. Contenido + tabla**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | TITLE | 0.60 | 0.60 | 12.13 | 1.35 |
| 1 | OBJECT | 0.61 | 2.10 | 3.19 | 4.70 |
| 13 | TABLE | 4.43 | 2.10 | 8.30 | 4.70 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |

**9. Dos contenidos**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | TITLE | 0.60 | 0.53 | 12.13 | 1.46 |
| 1 | OBJECT | 0.60 | 2.15 | 5.94 | 4.37 |
| 13 | OBJECT | 6.89 | 2.15 | 5.84 | 4.37 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |

**10. Tabla**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | TITLE | 0.60 | 0.61 | 12.13 | 1.30 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |
| 1 | OBJECT | 0.60 | 2.10 | 12.13 | 4.70 |

**11. Cierre**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | TITLE | 0.60 | 0.60 | 5.66 | 3.11 |
| 1 | OBJECT | 0.60 | 3.99 | 5.66 | 2.96 |
| 13 | PICTURE | 6.48 | 0.60 | 6.18 | 6.33 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |

**12. Diapositiva de título**

| idx | Type | Left | Top | Width | Height |
|---|---|---|---|---|---|
| 0 | CENTER_TITLE | 3.67 | 0.43 | 9.06 | 3.23 |
| 1 | SUBTITLE | 3.67 | 3.87 | 9.06 | 2.80 |
| 10 | DATE | 0.60 | 7.12 | 2.88 | 0.17 |
| 11 | FOOTER | 3.67 | 7.12 | 6.98 | 0.17 |
| 12 | SLIDE_NUMBER | 10.88 | 7.12 | 1.85 | 0.17 |

## Sample slides (to be removed in the deck)

| # | Layout |
|---|---|
| 1 | Solo el título |
| 2 | Agenda |
| 3 | Título de sección |
| 4 | Título + subtítulo + imagen |
| 5 | Título y contenido 1 |
| 6 | Título + subtítulo |
| 7 | Dos contenidos 1 |
| 8 | Dos contenidos 1 |
| 9 | Contenido + imagen |
| 10 | Contenido + tabla |
| 11 | Dos contenidos |
| 12 | Tabla |
| 13 | Cierre |

## How the deck uses this

- `src/intent/plot_style.py` draws figures on the master background (`#1B192E`) with `#FFFFFF` and `#A3A3C1` ink and sizes them to the placeholders above: full width 12.1 x 4.6 (layout 10), wide 8.3 x 4.7 (layout 8, right area), half 5.9 x 4.3 (layout 6, each column).
- Chart colors, in fixed order: `#109FAC`, `#AF5C4F`, `#D40AA8`, `#11AB7B` (teal, rose, magenta, green), plus `#5C5C85` for de-emphasis. They are the theme accents, snapped to the dark-surface colorblind checks; the reasoning and the validator results are in the module docstring.
- Layouts chosen per slide: `reports/slides_outline_es.md`.
