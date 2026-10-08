# Inspection of synthetic cases

These are labeled-data examples from the random development augmentation, not official test inputs. Source record ids are zero-based data-record indices after the supplied header, preserved before corrupted records are excluded.

## Same specialty does not ensure consistent sections

In `data/augmented/random_sections_development.csv`, an Ophthalmology variant of source **1110** has donors **1127** and **1143**. All three belong to the fitting partition and have the same supplied specialty.

The original title states: “Repair of entropion, left upper lid, with excision of anterior lamella and cryotherapy.” The synthetic title is retained. The `PREOPERATIVE DIAGNOSES` section is replaced with donor 1127's “Tearing, eyelash encrustation with probable tear duct obstruction bilateral.” The `PROCEDURE IN DETAIL` section comes from donor 1143 and describes entering the anterior chamber, capsulorrhexis and lens removal, including “The lens was then cracked with a McPherson forceps”.

The retained eyelid-repair title and substituted lens-removal procedure describe different operations. This demonstrates an observed inconsistency, even though the class label and fitting-fold provenance are preserved. Fold isolation prevents data leakage; it does not establish the clinical validity of generated text. The measured recombination gains are small, and the resampling control is slightly stronger in both diagnostic regimes.

## A second observed mismatch

A Neurosurgery variant of source **900**, with donors **907, 910, 915, 946**, retains a title describing phenol neurolysis and botulinum toxin injections. Its original preoperative diagnosis is “Spastic right hemiparetic cerebral palsy”; the variant replaces that with “Tremor, dystonic form” and its postoperative diagnosis with “To be determined after CSF evaluation.” The procedure codes also change. This is further evidence that arbitrary same-header exchanges can break cross-section consistency.

## Interpretation

Resampling changes how often existing minority cases are seen. Dropout changes which fields are available without adding clinical assertions. Section recombination adds lexical diversity but can combine incompatible diagnoses and procedures. A future version should preserve linked diagnosis/procedure sections or reject incompatible donor combinations, and should be evaluated again rather than assumed to improve classification.
