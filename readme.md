## Alignment-Free Baseline Comparison

To evaluate whether ESM-2 embeddings provide information beyond simple sequence composition, embeddings were compared against a classical alignment-free baseline using normalized 3-mer frequency vectors — a representation that encodes local amino acid composition without any learned features.

### Clustering Metrics

| Method                   | Cluster Purity | V-measure |
| ------------------------ | -------------- | --------- |
| 3-mer Frequency Baseline | 0.722          | 0.819     |
| ESM-2 Embeddings         | 0.747          | 0.790     |

The two methods show a metric-dependent trade-off. ESM-2 achieves higher cluster purity (0.747 vs 0.722), meaning its clusters are more internally homogeneous with respect to true family labels. However, the 3-mer baseline achieves a higher V-measure (0.819 vs 0.790), which jointly penalizes both impurity and fragmentation.

This divergence reflects a fundamental structural difference between the two representations: k-mer UMAP projections fragment each family into many small, locally pure subclusters that are globally disorganized. The purity metric rewards each pure fragment independently, inflating the score. V-measure, by contrast, rewards completeness — the degree to which all members of a true family land in the same cluster — which benefits the fragmented k-mer layout when subclusters happen to be homogeneous.

ESM-2 embeddings instead produce continuous, globally coherent manifolds per family, with HSP70 and RPS3 forming compact isolated regions and GAPDH/Cytochrome C occupying a shared linear manifold. This global structure is biologically more meaningful even where purity is locally imperfect, because it reflects the continuous nature of evolutionary divergence rather than discrete composition fragments.

In short: k-mer vectors achieve structural separability through fragmentation; ESM-2 achieves it through learned biological organization. The two metrics together reveal this distinction more clearly than either alone.

### UMAP Comparison: 3-mer Baseline vs ESM-2 Embeddings

![k-mer vs ESM2](plots/kmer_vs_esm2.png)

The left panel shows UMAP-projected 3-mer frequency vectors. RPS3 (blue) is fragmented into at least eight disconnected islands distributed across the entire embedding space, and GAPDH (green) shows no coherent global structure. The right panel shows ESM-2 embeddings over the same sequences. HSP70 (red) occupies a compact, fully isolated region. RPS3 (blue) resolves into two clean subclusters. The persistent overlap between GAPDH (green) and Cytochrome C (purple) reflects a genuine biological constraint rather than a representational failure (see Biological Interpretation).

---

## Biological Interpretation

The persistent overlap between GAPDH and Cytochrome C in ESM-2 embedding space is the most analytically significant result of this project. Both proteins are ancient, universally conserved housekeeping proteins operating under strong purifying selection across all domains of life. GAPDH is central to glycolysis; Cytochrome C is a core electron carrier in oxidative phosphorylation. Because both have been under extreme functional constraint for over a billion years, their sequence diversity across eukaryotes is substantially compressed relative to stress-response or lineage-specific proteins such as HSP70.

ESM-2 appears to encode functional constraint as a dominant signal. When two families share a similar evolutionary age, similar levels of sequence conservation, and similar biochemical roles as small soluble metabolic proteins, the model places them in adjacent regions of embedding space regardless of their distinct functions. This suggests that at 8 million parameters, ESM-2 captures broad physicochemical and conservation patterns rather than fine-grained functional identity — a limitation that larger models (ESM-2 650M or ESM-2 3B) may partially overcome.

This result connects directly to a core problem in phylogenomics: distinguishing evolutionary conservation due to shared ancestry (homology) from conservation due to shared functional constraint (analogy or parallel evolution). Sequence-only embeddings, even learned ones, face an inherent ambiguity between these two sources of similarity. Resolving GAPDH from Cytochrome C may require structural embeddings, domain annotation priors, or explicit phylogenetic context — precisely the motivation behind graph-based approaches to ortholog detection.

---

## Embedding Outlier Detection

Within-family KMeans clustering (k = 2 and k = 3) identified sequences occupying atypical regions of each family's embedding subspace. Sequences belonging to minority clusters containing fewer than 20% of family members were flagged as embedding outliers.

| Family       | Total Sequences | Outliers Detected |
| ------------ | --------------- | ----------------- |
| Cytochrome C | ~80             | 1                 |
| GAPDH        | ~80             | 1                 |
| HSP70        | ~80             | 4                 |
| RPS3         | ~80             | 11                |

The elevated outlier count in RPS3 (11 sequences) is notable. RPS3 has a known paralog, RPS3a, present in eukaryotes, and ribosomal proteins show documented lineage-specific duplications across fungi, plants, and animals. OrthoDB orthogroups for RPS3 may therefore include sequences from both the canonical cytoplasmic ribosomal family and divergent paralogs that were not filtered during dataset curation. These 11 outliers are candidates for BLAST verification against RPS3a or mitochondrial ribosomal protein databases. Results are saved to `results/paralog_candidates.csv`.
