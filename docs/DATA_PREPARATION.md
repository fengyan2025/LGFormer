# Data preparation and availability

The experiments used paired HCP T1-weighted structural MRI slices. Low-quality
inputs represent 2.0-mm-quality acquisitions and references represent
0.7-mm-reference-quality images on the same `304 x 256` grid. Images were
normalized to approximately `[-1, 1]`.

Raw HCP data and subject identifiers are not distributed. Users must obtain
authorized data independently and comply with the applicable HCP data-use
terms.

The frozen internal repartition contains 1,071 subjects and 21,420 image pairs:

| Split | Subjects | Pairs |
|---|---:|---:|
| Train | 857 | 17,140 |
| Validation | 107 | 2,140 |
| Test | 107 | 2,140 |

Splitting is subject-disjoint. All subjects appeared somewhere in older project
exploration; the Test set is an internal fresh repartition rather than an
untouched external cohort.
