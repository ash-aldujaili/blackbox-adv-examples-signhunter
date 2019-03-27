# blackbox-adv-examples-signhunter
A repository for the query-efficient black-box attack, SignHunter: **[There are No Bit Parts for Sign Bits in Black-Box Attacks](https://arxiv.org/abs/1902.06894v1)** by Abdullah Al-Dujaili and Una-May O'Reilly, 2019


**Highlights of the attack**:
- 100% evasion rate on 10K MNIST images using 12 queries on average!
- 90% evasion rate against Ensemble Adv. trained model for IMANGENET with 1000 queries!
- 2.5x fewer queries and 3.8x less failure-prone that SOTAs combined on MNIST, CIFAR10, & IMAGENET.
- Theoretically guaranteed to perform at least as well as FGSM after 2^(log(n)+1) queries.
- Surpasses all the transfer-only zero-query attacks on Madry's Lab's challenges and is comparable with PGD using the same number of iterations with the difference that PGD uses X backprops, while SignHunter uses X forwards.


### Repository Structure

The repository is of the following structure:
- `src/`: contains the code for reproducing our work.
- `data/`: a placeholder for the data files that were used to generate our figures. These can be downloaded upon request. Please fill the form at <https://goo.gl/forms/08BYk1Ex49wg5chu2>. Note that these data files are not required for running the code. You can pretty much regenerate the data on your own.

Refer to the directories above for code & data setups.

Also, the repository has two branches:
- `master`: contains the code for the main experiments in the paper.
- `chanllenges`: for experiments of Section 6.

### More Resources:

- Coming Soon


### Citation

If you make use of this code and you'd like to cite us, please consider the following:

```
@article{al2019there,
  title={There are No Bit Parts for Sign Bits in Black-Box Attacks},
  author={Al-Dujaili, Abdullah and O'Reilly, Una-May},
  journal={arXiv preprint arXiv:1902.06894},
  year={2019}
}
```

### Acknowledgement 

This work was supported by the MIT-IBM Watson AI Lab.
The authors would like to thank Shashank Srikant for his
timely help.
