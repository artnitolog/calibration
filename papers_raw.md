<!-- python -m readme2tex --output papers.md --nocdn --rerender papers_raw.md -->
# Конспекты статей


why to use logistic transform all the time? the curve may be skewed etc

## Niculescu-Mizil, A., & Caruana, R. [Predicting good probabilities with supervised learning](https://sci-hub.do/10.1145/1102351.1102430), 2005
* Authors compare (mis)calibration of popular ML algorithms (tree-based, naive bayes, logregression, and SVM)
* Distribution of probability estimates: bell-shaped (boosted tree-based, SVM) vs U-shaped (NB, RF, logistic).
* Tendencies in reliability plots: sigmoid-shaped (SVM, trees) —    underconfidence, «inverse sigmoid»-shaped (NB) — overconfidence
  * Symmetric unimodal shape is typical for max-margin methods like SVM or boosted trees — mass is in the center. In case of imbalanced classes the situation is different...
  * In NB peaks of U are at 0 and 1 because of unrealistic assumtion (conditional independance of features given class)
  * In RF peaks of U are slightly pushed from 0 and 1 due to high variance of individual trees
  * In logistic regression (and other models afrer calibration) the distibution is U-shaped with peaks close to 0 and 1
* 2 calibration methods are used: Platt scaling & isotonic regression. Platt scaling is good when calibration curve is sigmoidal (like in SVM) or size of 
* In fact, all sklearn examples on calibration exploit only this paper

## R. Müller, S. Kornblith, G. Hinton [When Does Label Smoothing Help?](https://arxiv.org/pdf/1906.02629.pdf), 2019

> **General idea (introduced in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf))**: replace the hard (true) targets $y_k$ in standard cross-entropy
> $$H(y, p) =-\sum_{k=1}^K y_k \log{p_k}$$
> with smoothed (mixture) $y_k^\prime = (1-\alpha)y_k +\alpha u_k$. The prior distribution over labels $u_k$ is now taken uniform: $u_k = \frac{1}{K}$. This regularization mechanism was called LSR (label-smoothing regularization).
> The final loss:
> $$H(y^\prime,p)=-\sum_{k=1}^K ( (1-\alpha)y_k +\frac{\alpha}{K})\log{p_k} \\
=(1-\alpha)H(y, p)+\alpha H(u, p)$$
> **So what?** The model is encouraged to be less confident. LSR prevent the largest logit from becoming much larger than the other. In fact, we reduce overfitting.

In the [paper]((https://arxiv.org/pdf/1906.02629.pdf)), the authors explain why this trick works on model calibration (and against distillation).

Intuitively, $\lVert x-w_k\rVert^2 = \lVert x \rVert ^2 + \lVert w_k \rVert ^2- 2x^T w_k$, so each class has a template $w_k$; $\lVert x \rVert ^2$ does not depend on the class, $\lVert w_k \rVert ^2$ is usually constant across classes. So the logit $x^T w_k$ can be thought of as a measure of distance between the penultimate layer and a template.

**Insight**: label smoothing encourages the activations $x$ of the penultimate layer to be close to the template of the correct class $w_k$ and equally distant to the templates of the incorrect class. This property is observed in authors' visualizations on the datasets CIFAR-10, CIFAR-100 and ImageNet with the architectures AlexNet, ResNet-56 and Inception-v4.

**Label smoothing prevents over-confidence. But does it improve the calibration of the model?**
The authors show that label smoothing reduces ECE and can be effective even without temperature scaling.

## Chuan Guo, Geoff Pleiss, [«On Calibration of Modern Neural Networks»](https://arxiv.org/abs/1706.04599), 2017

Authors explain basic concepts of calibration and compare different methods on real models and datasets. The main result is coolness of temperature scaling in all respects.

### Intro

* In «real-world», classification with high accuracy isn't always enough. In addition to class label, a network is expected to provide a *calibrated confidence* to its prediciton, which means that the score of the class should reflect the real probability. Motivation:
    1. Decision systems like self-driving cars or automated disease diagnosis.
    2. Model interpretability.
    3. Good probability estimates may be incorporated as inputs to other models.
* Modern neural networks are more accurate than they used to be, but less calibrated. Better accuracy doesn't mush better confidence.

### How to estimate (mis)calibration

* Perfect calibration: $\mathbb{P} \left(\widehat{y}=y\mid\widehat{p}=p\right)=p,\quad \forall p\in\left[0,1\right]$.
* Reliability histogram — accuracy as a function of confidence.
  1. We split confidences $\widehat{p}_1,\dots,\widehat{p}_n$ into bins (by size or length) $B_1,\dots,B_M$.
  2. $\operatorname{acc}(B_m)=\frac{1}{|B_m|}\sum_{\widehat{p}_i\in B_m}\left[\widehat{y}_i=y_i\right]$ (unbiased and consistent estimator of $\mathbb{P} \left(\widehat{y}=y\mid\widehat{p}\in B_m\right)$).
  3. $\operatorname{conf}(B_m)=\frac{1}{|B_m|}\sum_{\widehat{p}_i\in B_m}{p}_i$ — average confidence.
* Miscalibration measure is $\mathbb{E}_{\widehat{p}}\lvert\mathbb{P} \left(\widehat{y}=y\mid\widehat{p}=p\right)-p\rvert$. To estimate this value, ECE (expected calibration error) is used:
$$\operatorname{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}
\underbrace{\lvert \operatorname{acc}(B_m)-\operatorname{conf}(B_m)\rvert}_\text{gap in reliability diargram}$$

* Sometimes we may wish to estimate only worst-case deviation (expectation is replaced with max). Approximation is MCE (maximum calibration error) — like ECE, but sum is replaced with max.
* Also NLL (negative log likelihood) as a standard measure of a probabilistic model’s quality can be used to estimate calibration.

### Reasons of miscalibration

1. **Model capacity.** Experiments showed that ECE metric grows substantially with increasing depth, filters per layer. But subjectively results seem incomplete.
2. **Batch normalization.**  Authors claim that models trained with Batch Normalization tend tobe more miscalibrated.
3. **Weight decay.** It is common to train models with little weight decay, but calibration is improved when more regularization is added.
4. **NLL**. Overfitting to NLL can be benificial to classification accuracy. At the expense of well-modeled probability.

### Calibration methods

#### Binary classification

* **Histogram binning.** We split (by size or length) uncalibrated confidences into bins and assign a substitute to each one. These substitutes are chosen to minimize bin-wise square loss.
* **Isotonic regression** — generalization of histogram binning: we optimize not only substitutes, but also bin boundaries. The loss function is the same. Optimization is now constrained with condition of monotone boundaries and substitutes.
* **Bayesian binning into quantiles.** Instead of producing a single binning scheme, BBQ performs Bayesian averaging of the probabilities produced by each sheme.
* **Platt scaling.** Uncalibrated predictions are passed through trained sigmoid (shift and scale). In fact, it is a logistic regression with uncalibrated confidences as inputs.

#### Multiclass

* **Binning methods** can be incorparated with one-vs-all method.
* **Matrix scaling.** Before passing to softmax, we apply linear transformation to the logits. This linear transformation may be restricted to coordinate scaling (*vector scaling*) and even a single scalar parameter:
* **Temperature scaling.** Logits are only scaled with *temperature* T. Classification is not affected with this transormation. T is optimized (like Platt scaling) with respect to NLL on validation set (can this be done with folding?)

#### Experiment results

* Authors used state-of-the-art (2017) NN models and different datasets (image classification and document classification).
* The main result is *surprising* effectiveness of temperature scaling. Vector scaling after fitting produced nearly constant parameters, which made it no different than a scalar transformation. So, authors concluded «network miscalibration is intrinsically low dimensional».
* The concept «simple is better than complex» worked in histogram methods too: just histogram binning outperformed BBQ and isotonic regression.
* As for computation time, one-parameter temperature scaling is the fastest one. It is followed by histogram methods.

## J. Platt. [«Probabilistic outputs for support vector machines and comparison to regularized likelihood method»](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639), 2000

* The general idea of getting conditional probabilities of classes is to make post-processing step: algorithm outputs (margins) are transformed with fitted sigmoid (2 parameters)
* Parameters can be found with MLE. The issues are the choice of calibration set and the method to avoid overfitting
* Using the same set (whole train) to fit the sigmoid can give extremely biased fits, so special hold-out set is preferred (cross-validation will also five a lower variance estimate for sigmoid parameters)
* Overfitting of sigmoid is still possible. Suggested methods:
    1. Regularization (requires a prior model for parameters)
    2. Using smoothed targets (for MAP estimates) instead of {0, 1}

## Zadrozny В., Elkan C. [«Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers»](https://cseweb.ucsd.edu/~elkan/calibrated.pdf), 2001

* Applying **m-estimation** (generalized Laplace smoothing) to correct most distant probabilities and shift towards the base rate(standard one, based on [rule of succession](https://en.wikipedia.org/wiki/Rule_of_succession) adjusts estimates closer to 1/2 which is not reasonable in unbalanced classes). [details](https://www.researchgate.net/publication/220838515_Estimating_Probabilities_A_Crucial_Task_in_Machine_Learning)
* \[DT\] **Curtailment** - taking into account not only class frequencies of the leaves but also of the closest ancestors. can be implemented with unconventional pruning
* **Binning** (histogram method): training examples are sorted by scores into subsets of equal size, and the _estimated corrected probability_ for the test object is now the "accuracy" inside the bin it belongs to
* **Evaluation metrics** of calibration quality: MSE and log-loss are more suitable than lift charts and "profit achieved" (specific problem-related metric)

* DTs' _predict proba_ is just the raw training frequency of final leaf. That's not reliable:
    1. such frequencies are usually shifted towards 0 or 1 since DTs strive to have homogeneous leaves;
    2. without pruning, leaf «capacity» can be small
* Standard DT pruning (maximizing accuracy) methods do not improve quality of probability estimation
