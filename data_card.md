
# Antibody Domainbed (Ab Db)

The Antibody Domainbed dataset has been designed to
1. incorporate state-of-the-art (SOTA) approaches in domain generalization to build robust predictors for the binding properties of biological sequences
2. assess domain generalization algorithms in a real-world problem inspired by therapeutic antibody discovery.

The dataset (sequence + structure) and corresponding backbones have been seamlessly integrated within the largest out-of-distribution (OOD) benchmark - Domainbed.

## Dataset Link


[Zenodo](https://zenodo.org/records/11446107)

## Data Card Author(s)

-   **Natasa Tagasovska, Prescient/MLDD, Genentech** (Owner /
    Contributor / Manager)
-   **Ji Won Park, Prescient/MLDD, Genentech** (Owner / Contributor /
    Manager)
-   **Matthieu Kirchmayer, Prescient/MLDD, Genentech** (Owner /
    Contributor / Manager)

## Authorship and Publishers

#### Publishing Organization(s)

Genentech

#### Industry Type(s)

Corporate (Biotech, Pharma)

#### Contact Detail(s)


-   **Publishing POC:** Natasa Tagasovska, Ji Won Park
-   **Affiliation:** Prescient/MLDD, Genentech
-   **Contact:**
    [natasa.tagasovska\@roche.com](mailto:natasa.tagasovska@roche.com),
    [jiwon.park\@gene.com](mailto:jiwon.park@gene.com)

### Dataset Owners

#### Team(s)

Genentech, Roche


#### Author(s)


-  Natasa Tagasovska, Principal ML Scientist, Prescient Design, Genentech, 2024
-  Ji Won Park, Principal ML Scientist, Prescient Design, Genentech, 2024
-  Matthieu Kirchmayer, Senior ML Scientist, Prescient Design, Genentech, 2024
-  Nathan C. Frey, Principal ML Scientist, Prescient Design, Genentech, 2024
-  Andrew Martin Watkins, Director of computational biology, Prescient Design, Genentech, 2024
-  Aya Abdelsalam Ismail, Senior ML Scientist, Prescient Design, Genentech, 2024
-  Arian Rokkum Jamasb, Senior ML Scientist, Prescient Design, Genentech, 2024
-  Edith Lee, Data Engineer, Prescient Design, Genentech, 2024
-  Tyler Bryson, Data Engineer, Prescient Design, Genentech, 2024
-  Stephen Ra, Director of Frontier Research, Prescient Design, Genentech, 2024
-  Kyunghyun Cho, Senior Director of Frontier Research, Prescient/MLDD, Genentech and NYU Center for Data Science, 2024


### Funding Sources

#### Institution(s)

-   Genentech

## Dataset Overview

#### Data Subject(s)

-  Synthetically generated antibodies are represented in the corresponding sequence and structure format.


#### Dataset Snapshot

| Category                     | Data   |
|------------------------------|--------|
| Size of Dataset              | 12.9 MB|
| Number of Instances          | 10 669 |
| Number of Fields             | 14     |
| Labeled Classes              | 2      |
| Number of Labels             | 10 669 |
| Number of environments       | 5      |
| Number of genrative models   | 5      |
| Number of wild types (seeds) | 179    |
| Number of antigens           | 3      |



#### Content Description

A given molecule has good therapeutic properties if it can tightly bind
to an Antigen (target) of interest. Since wet-lab binding measurements
are expensive (both in terms of money and time), domain experts rely on
computational properties such as Gibbs energy which has been shown to be
indicative of binding. We use published, ML sota models for generating a
library of in-vitro functional synthetic antibodies. We condition the
generation with an initial wild type, i.e. seeds, which were
selected from an existing database of experimentally obtained
antibody-antigen complexes. Then, each record in our dataset corresponds
to the difference in energy as computed by Rosetta scoring function ref2015, between the
original Ab-Ag complex and the AbMutated-Ag complex. The difference in
energy can be interpreted as 'decrease in energy' signaling
better/stronger binding of the two molecules. For each new design in our
database we record the source (generative model), antigen, heavy and
light chain with both sequence and structure representation. We split
the datapoints into environments emulating active drug design cycles.
The environment labels are intended to help DG algorithms and they can
also be modified by users of the datasets to mimic a setup corresponding
to their needs.


### Sensitivity of Data

-   this is a synthetic sequence library, there is no sensitive data.


#### Risk Type(s)

-   No Known Risks

### Dataset Version and Maintenance

#### Maintenance Status

**Actively Maintained** - No new versions will be made available, however,
this dataset will be actively maintained, including but not limited to
updates to the data and extending with new data representations.


#### Version Details


**Current Version:** 2.0

**Last Updated:** 05/2024

**Release Date:** 05/2024

#### Maintenance Plan

<!-- scope: microscope -->

**Versioning:** New versions will include adding new instances, new data transformations
or new data representations.

**Updates:** Updates may include addiitonal meta data or documentation.

**Errors:** Any reported errors, if affecing the dataset, will result with and update and new version.


## Example of Data Points

#### Primary Data Modality


-   Sequence
-   pdb structures (adjusting the original SabDab  complexes by applying InterfaceAnalyzerMover from Rosetta).


#### Data Fields


| Field Name    | Field Value                        | Description                                                                           |
|------------------------|------------------------|------------------------|
| seq id        | a07434d97d68b31f55da2efa851e77c6 | a unique design identifier                                                            |
| FV Heavy      | QVRLAQYGGGVKRLGASMTLSCVASGYTFNDYYIHWVRQAPGQGFELLGYIDPANGRPDYAGALRERLSFYRDKSMETLYMDLRSLRYDDTAMYYCVRNVGTAGSLLHYDHWGSGSPVIVSS | variable framework region, heavy chain sequence                                       |
| FV Light      | EIVLTQSPATLSASPGERVTLTCKASRSVGNNVAWYQHKGGQSPRLLIYDASTRAAGVPARFSGSASGTEFTLAISNLESEDFTVYFCLQYNNWWTFGQGTRVDIK | variable framework region light chain sequence                                       |
| FV Heavy aHo  | QVRLAQY-GGGVKRLGASMTLSCVASG-YTFND-----YYIHWVRQAPGQGFELLGYIDPA---NGRPDYAGALRERLSFYRDKSMETLYMDLRSLRYDDTAMYYCVRNVGTAG-----------------SLLHYDHWGSGSPVIVSS | heavy chain sequence after aHo alignment                                              |
| FV Light aHo  | EIVLTQSPATLSASPGERVTLTCKAS--RSVG------NNVAWYQHKGGQSPRLLIYD--------ASTRAAGVPARFSGSASG--TEFTLAISNLESEDFTVYFCLQYNN------------------------ | light chain sequence after aHo alignment                                              |
| pdb           | 2adf                               | pdf identifier of initial Ab-Ag complex                                               |
| ddG           | 2.49                             | difference in energies of the initial complex and the complex after mutations have been applied   |
| is_more_stable | 0                                  | binary label stating if the designed complex is more stable (indicating better binding) than the initial complex or not |
| source        | WJS 0.5                            | generative model for the design - walk jump sampler with noise level 0.5               |
| antigen       | HIV1                               | antigen target of interest                                                            |
| environment   | env0                               | categorical variable representing the environemnet (design cycle)                     |
| edist seed    | 4                           | edit distance between initial antibody (seed) and proposed design                                     |

**Above:** Example of an actual data point with descriptors

## Motivations & Intentions

### Motivations

#### Purpose(s)


-   Research
-   ML for drug design
-   DG algorithms in science applications

#### Domain(s) of Application


`Out-of-distribution Generalization`, `Distribution shift`,
`Drug design`,`proteins`, `antibodies`.


#### Motivating Factor(s)

-   ML models are prone to leach on spurious correlations to maxmize
    performance, which leads to poor results when the same models
    are faced with OOD data. DG could help aleviate this issue.
-   DG algorithms have been tested mostly on image datasets and in part,
    this might be due to the gap between DG algorithms and science
    datasets. This is the gap we aim to fill with Antibody Domainbed.

### Intended Use


#### Suitable Use Case(s)

- Evaluating existing and new DG algorithms.

- Evaluating existing and new protein property
predictors.

-Pretraining/Fine-tuning property predictors on
Antibody Domainbed.

- Custom split into new set of environments to
evaluate environment discovery algorithms.

- Extending the dataset by computing additional
properties for the antibodies (hydrophobicity, charge, humanness etc).


#### Unsuitable Use Case(s)

- None of the included molecules have been tested
in wet-lab experiments, hence they are not advised to be used as
therapeutic proteins without consulting domain experts.

- It is not advised to use the included pdbs of the synthetic designs for
computing the impact on energy by further mutations, due to error
propagation. Please use the initial/seed pdb files if applying new
mutations to the datapoints in Antibody domainbed.


#### Research and Problem Space(s)


- Can we learn invariant/causal features that drive the binding of
molecular complexes?
-  Do DG algorithms pick up physically meaningful
features?
- Does heterogenous data help in developing more roust
predictors?
- Can we trust ML models for predicting properties for
new/unseen targtes?

#### Citation Guidelines

<!-- scope: microscope -->
Citation: Tagasovska, N., Park, J. W., Kirchmeyer, M., Frey, N., Watkins, A.,
Abdelsalam Ismail, A., Jamasb, A. R., Lee, E., Bryson, T., Ra, S., & Cho, K.
(2024). Antibody DomainBed: Out-of-Distribution Generalization in Therapeutic Protein Design.
https://doi.org/10.5281/zenodo.11446107

**BiBTeX:**

@article{abdb2024,
  title={Antibody DomainBed: Out-of-Distribution Generalization in Therapeutic Protein Design},
  author={Tagasovska, N., Park, J. W., Kirchmeyer, M., Frey, N., Watkins, A.,
Abdelsalam Ismail, A., Jamasb, A. R., Lee, E., Bryson, T., Ra, S., Cho, K},
  journal={in submission},
  year={2024},
}

## Access, Rentention

### Access

-   External - Open Access
Antibody Domainbed is made available under the CC BY 4.0 license. A copy of the license is provided
with the dataset. The authors bear all responsibility in case of violation of rights.

#### Documentation Link(s)


-   [Github repo with code](https://github.com/prescient-design/antibody-domainbed)
-   [Croissant meta-data](https://github.com/prescient-design/antibody-domainbed)

#### Prerequisite(s)

-   No prerequisits for using the dataset.



## Provenance

### Collection

#### Method(s) Used

-   Artificially Generated Dataset

The primary goal of this benchmark is to emulate a real-world active drug design setup. To create a realistic and publicly accessible dataset, we followed these steps: (1) Collect open data on antibody-antigen complex structures; (2) Train generative models to sample antibodies with varying statistics (e.g., edit distances from training data, targets of interest, different seed complexes); (3) Compute binding proxies using physics-based models for all designs from step (2); and (4) Split the labeled dataset into meaningful environments for the application of domain generalization (DG) algorithms. In the full manuscript, we show that our antibodies span reasonable ranges in various metrics, including biophysical properties (e.g., hydrophobicity, aromaticity) and distributional scores evaluating proximity to a database of natural antibodies (e.g., naturalness).


#### Step 1: Data curation ####

We select antibodies associated with the antigens HIV1, SARS‑CoV‑2, and HER2. For Environments (Env) 0-3, we post-process paired antibody-antigen structures from the latest version (released in 2022) of the Structural Antibody Database (SAbDab). Antibody sequences corresponding to 178 SAbDab structures for the three antigens serve as the starting seeds for our sequence-based generative models (see Step 2), and we modify the structures based on mutations made by the model to compute the labels (see Step 3). For Env 4, we use a recent derivative, Graphinity, which extends SAbDab to a synthetic dataset of a much larger scale (~1M) by introducing systematic point mutations in the CDR3 loops of the antibodies in the SAbDab complexes.

#### Step 2: Sampling antibody candidates ####
 To emulate the active drug discovery pipeline, we need a suite of generative models for sampling new candidate designs for therapeutic antibodies. We run the Walk Jump Sampler, a method building on the neural empirical Bayes framework.
WJS separately trains score- and energy-based models to learn noisy data distributions and sample discrete data. The energy-based model is trained on noisy samples, which means that by training with different noise levels $\sigma$, we obtain different generative models. Higher $\sigma$ corresponds to greater diversity in the samples and higher distances from the starting seed. We used four values for the noise parameter, namely $\sigma \in \{0.5, 1.0, 1.5, 2.0\}$.

#### Step 3: Labeling candidates ####
As lab assays to experimentally measure binding affinity are prohibitively expensive, we use computational frameworks which, by modeling changes in binding free energies upon mutation (interface $\Delta\Delta G = \Delta G_\textrm{wild type} - \Delta G_{\rm mutant}$), enable large-scale prediction and perturbation of protein–protein interactions. We use the \verb|pyrosetta| (pyR) implementation of the \verb|InterfaceAnalyzerMover|, namely the scoring function \verb |ref2015|, to compute the difference in energies before and after mutations in a given antibody-antigen complex. After removing highly uncertain labels between -0.1 and 0.1 kcal/mol, we attach binary labels to each candidate of the generative models: label 1 if $\Delta\Delta G < -0.1$ (stabilizing) and label 0 if $\Delta\Delta G > 0.1$ (destabilizing). While computed $\Delta\Delta G$ represent weak proxies of binding free energy, they have been shown to be predictive of experimental binding. See the full manuscript for details.

#### Step 4: Splitting into environments ####
To emulate the active drug design setup where distribution shifts may appear in each design cycle (e.g., due to new generative models, antigen targets, or experimental assays), we split the overall dataset into five environments. In Table 1 in the manuscript we summarize our environment definitions as well as key summary statistics of each environment. For Env 0-3, the dataset split mimics sub-population shifts due to the generative model, which produces antibody sequences with different edit distances from the seed sequences. The WJS model with $\sigma{\rm =}0.5$ ($\sigma{\rm =}2.0$) produces antibody designs close to (far from) the seed. Env 4 has been partially included in the experiments because it introduces severe distribution shift in the form of concept drift and label shift, as it represents a completely new target and a different labeling mechanism than the rest. We report preliminary results in this extreme setup in the appendix.

**Is this source considered sensitive or high-risk?** [No]

**Dates of Collection:** [07 2023 - 12 2023]

**Primary modality of collection data:**


-  Therapeutic protein sequences and structures.

#### Source Description(s)

-   **Source:** Structural Antibody Database, SAbDab published in 2022.
-   **Source:** Grafinity - Antibody-Antigen ∆∆G Prediction 2023.

We used these two sources to curate the set of antibodies for training our generative models and use some hold out datasets to represet initial points, i.e. seeds.



#### Limitation(s) and Trade-Off(s)


- The structures used in the computation of changes in binding energy were derived using computational tools rather than experimentally measured structures, which are very expensive to obtain. Hence, our dataset does include some biases inherent to the computational methods.

- We don't include the structures for env4, however, they can be found in the original repository of the Grafinity dataset.

### Version and Maintenance

#### First Version

<!-- scope: periscope -->

-   **Release date:** 05/2024
-   **Link to dataset:** [Zenodo](https://zenodo.org/records/11401578) version 1
-   **Status:** [Actively Maintained]
-   **Size of Dataset:** 12.9 MB (sequence);
-   **Number of Instances:** 10 669

#### Second Version

<!-- scope: periscope -->

-   **Release date:** 05/2024
-   **Link to dataset:** [Zenodo](https://zenodo.org/records/11446107) version 2
-   **Status:** [Actively Maintained]
-   **Size of Dataset:** 12.9 MB (sequence); 1.9 GB (structure)
-   **Number of Instances:** 10 669
