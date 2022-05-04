# Paired Permutation Test
This library contains implementations for an exact and efficient algorithm to
execute the paired-permutation test. Specifically, the library contains command-line
tools to run the test in the case of a difference of accuracy and a difference in F1 scores.
This library is the code-base for ["Exact Paired-Permutation Testing for Structured Test Statistics"](https://arxiv.org/abs/2205.01416).

## Citation

This code is for the paper _Exact Paired-Permutation Testing for Structured Test Statistics_ featured in NAACL 20222.
Please cite as:

```bibtex
@inproceedings{zmigrod-etal-2022-exact,
    title = "Exact Paired-Permutation Testing for Structured Test Statisticse",
    author = "Zmigrod, Ran  and
      Vieira, Tim  and
      Cotterell, Ryan",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    url = "https://arxiv.org/abs/2205.01416",
}
```

## Requirements and Installation

* Python version >= 3.6
* Numba

Installation:
```bash
git clone https://github.com/rycolab/paired-perm-test
cd paired-perm-test
pip install -e .
```

## Running a Paired-permutation Test
The command to run a paired permutation is given below
```bash
python pairedpermtest/run.py --input [input_file] --metric [acc/f1] --MC [K] --delim [,/tab]
```
`--input` is the input file formatted as explained below. 

`--metric` selects your test statistic.
Currently we offer a difference in accuracy (`acc`) and a difference in F1 scores (`F1`).
NOTE: the permutation test on a difference in F1 scores takes some time for large inputs.
Please only use on small datasets.

`--MC` is an optional argument to run a Monte Carlo approximation
instead of an exact algorithm. The argument to this is the
number of samples you would like to run the MC approximation with.

`--delim` indicates the delimmiter for the input file.
This can either be `,` for a csv file or `tab` for a tsv file.

### Input files
We currently support paired-permutation tests for a difference
in accuracy or F1 scores between two systems.
Input files are either `.csv` or `.tsv` files where each
line represents the metrics of one system.

For accuracy, please give the number of correct predictions
from system 1 and then the number of correct predictions from system 2.
These should be integers.
For example:
```
8,6
3,5
7,7
```
is the input file to compare the accuracies of two systems over three sentences where
system 1 has 8, 3, and 7 correct predictions for the three sentences
and
system 2 has 6, 5, and 7 correct predictions for the three sentences.

For F1, please give the number of true positive and incorrect predictions
from system 1 and then the number of true positive and incorrect predictions from system 2.
These should be integers.
For example:
```
8,2,6,1
3,3,5,2
7,1,7,1
```
is the input file to compare the F1 scores of two systems over three sentences where
system 1 predicted 8, 3, and 7 true positive predictions
in the three sentences and 2, 3, and 1 incorrect predictions
in the tree sentences.
Additionally, 
system 2 predicted 6, 5, and 7 true positive predictions
in the three sentences and 1, 2, and 1 incorrect predictions
in the tree sentences.