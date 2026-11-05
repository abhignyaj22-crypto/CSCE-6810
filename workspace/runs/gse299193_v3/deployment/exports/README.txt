Exports of the morphology-only program maps.

<slide>_qupath.tsv          tab separated; x_um, y_um and one 'Measurement: <target>_pred' column per target;
                            import in QuPath as detections (File > Import objects).
<slide>_xenium_explorer.csv cell_id, x_centroid, y_centroid (microns) and one column per target;
                            load in Xenium Explorer as a cell group or feature table.
<target>_pred               predicted standardised program score (cohort z units of the training run)
<target>_std                predictive standard deviation (temperature-scaled heteroscedastic head), when present
lineage_prob_<name>         composition-head probability of each lineage, when present

Scores are gene-set abundances associated with drug resistance in the literature, not pathway
activity and not a drug-response measurement (docs/research/PATHWAY_INTERPRETATION.md).
