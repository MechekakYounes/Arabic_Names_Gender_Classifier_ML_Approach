Arabic Name Gender Classifier

This project predicts gender (male / female) based on Arabic first names using a machine learning model.
It includes training, testing, and usage scripts to apply the model on Excel files containing names.

Main dependencies:

pandas
scikit-learn
openpyxl

Notes:

Only works reliably with Arabic names.
Handles cases where names are missing or invalid.
If input contains numbers or English names, they will be ignored/skipped.
