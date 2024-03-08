#!/usr/bin/env bash
set -ex

curl -sSL https://github.com/imantsm/medical_abbreviations/archive/4f1f4afefe0efc4b37fed43c8482d49ad87b0706.zip -o medical_abbreviations.zip
unzip -q medical_abbreviations.zip
cd medical_abbreviations-4f1f4afefe0efc4b37fed43c8482d49ad87b0706/CSVs
awk '(NR == 1) || (FNR > 1)' *.csv > medical_abbreviations.csv
mv medical_abbreviations.csv ../../data/raw
cd ../../
rm -fr medical_abbreviations.zip medical_abbreviations-4f1f4afefe0efc4b37fed43c8482d49ad87b0706
