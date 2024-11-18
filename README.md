Unofficial PyTorch version of ProteInfer, originally implemented in TensorFlow 1.X and converted for PyTorch compatibility.




To transform ProteInfer original tfrecords dataset into Fasta's, run the following command:

```
conda env create -f proteinfer_conda_requirements.yml
conda activate proteinfer
python bin/make_proteinfer_dataset.py --data-dir data/clustered_split/ --annotation-types GO
python bin/make_proteinfer_dataset.py --data-dir data/clustered_split/ --annotation-types EC
python bin/make_proteinfer_dataset.py --data-dir data/random_split/ --annotation-types GO
python bin/make_proteinfer_dataset.py --data-dir data/random_split/ --annotation-types EC
```