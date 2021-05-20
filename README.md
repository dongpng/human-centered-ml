
# Fairness

## Introduction
### Overviews
* Algorithmic Fairness: Choices, Assumptions, and Definitions, Mitchell et al. ,  Annual Review of Statistics and Its Application 2021 [[url]](https://www.annualreviews.org/doi/abs/10.1146/annurev-statistics-042720-125902?journalCode=statistics)
* Online book: Fairness and machine learning, Limitations and Opportunities, Barocas, Hardt and Narayanan https://fairmlbook.org/

### Types of harms
* Talk by Kate Crawford "The trouble with bias" 2017 [[Youtube]](https://www.youtube.com/watch?v=fMym_BKWQzk)

### Data

**Bias**

* Statistical bias vs. societal bias, see Algorithmic Fairness: Choices, Assumptions, and Definitions, Mitchell et al.,  Annual Review of Statistics and Its Application 2021 [[url]](https://www.annualreviews.org/doi/abs/10.1146/annurev-statistics-042720-125902?journalCode=statistics)
* Datasets from “Patterns, Predictions, and Actions“ by Hardt and Recht, 2021 [[url]](https://mlstory.org/data.html)
* Selection bias: No Classification without Representation: Assessing Geodiversity Issues in Open Data Sets for the Developing World, Shankar et al, NIPS 2017 workshop: Machine Learning for the Developing World [[url]](https://research.google/pubs/pub46553/)
* Annotator bias: The Risk of Racial Bias in Hate Speech Detection, Sap et al., ACL 2019 [[url]](https://www.aclweb.org/anthology/P19-1163/)
* Problematic labels: Dissecting racial bias in an algorithm used to manage the health of populations, Obermeyer et al., Science 2019 [[url]](https://science.sciencemag.org/content/366/6464/447)

**Ethics**

*Data collection*

* "The Internet Is Enabling a New Kind of Poorly Paid Hell", The Atlantic Jan 23, 2018 [[url]](https://www.theatlantic.com/business/archive/2018/01/amazon-mechanical-turk/551192/)
* "The Trauma Floor: The secret lives of Facebook moderators in America", The Verge Feb 25, 2019 [[url]](https://www.theverge.com/2019/2/25/18229714/cognizant-facebook-content-moderator-interviews-trauma-working-conditions-arizona)

*Documentation*

* Datasheets for Datasets, Gebru et al. 2018 [[url]](https://arxiv.org/pdf/1803.09010.pdf)
* Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science, Bender and Friedman, TACL 2018 [[url]](https://www.aclweb.org/anthology/Q18-1041/)

### Model development 

* Bias amplification: Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints, Zhao et al., EMNLP 2017 [[url]](https://www.aclweb.org/anthology/D17-1323/)
* Accuracy vs. fairness:  see chapter in "The Ethical Algorithm: The Science of Socially Aware Algorithm Design" by Roth and Kearns
* Example: Algorithmic Bias? An Empirical Study of Apparent Gender-Based Discrimination in the Display of STEM Career Ads. Lambrecht and Tucker, Management Science 65(7):2966-2981, 2019 [[url]](https://pubsonline.informs.org/doi/10.1287/mnsc.2018.3093)

*Documentation*

* Model Cards for Model Reporting, Mitchell et al. 2018 [[url]](https://arxiv.org/abs/1810.03993), see also this [online Google example](https://modelcards.withgoogle.com/face-detection)

## Measuring Fairness

### Fairness in classification: Groups
**Historical context**

* 50 Years of Test (Un)fairness: Lessons for Machine Learning,  Hutchinson and Mitchell, FAT<sup>∗</sup> 2019 [[url]](https://dl.acm.org/doi/10.1145/3287560.3287600)
* Chapter 5 (esp part 1) of fairmlbook "Testing Discrimination in Practice" [[url]](https://fairmlbook.org/testing.html)

**Legal context**

* Bias Preservation in Machine Learning: The Legality of Fairness Metrics Under EU Non-Discrimination Law, Wachter et al., West Virginia Law Review, Forthcoming [[url]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3792772)

**Impossibilities**

* Fair prediction with disparate impact: A study of bias in recidivism prediction instruments, Chouldechova, Big Data, Special issue on Social and Technical Trade-Offs, 2017 [[url]](https://arxiv.org/abs/1703.00056)
* Inherent Trade-Offs in the Fair Determination of Risk Scores, Kleinberg et al., Innovations in Theoretical Computer Science (ITCS), 2017 [[url]](https://drops.dagstuhl.de/opus/volltexte/2017/8156/pdf/LIPIcs-ITCS-2017-43.pdf)
* The (Im)possibility of Fairness: Different Value Systems Require Different Mechanisms For Fair Decision Making, Friedler et al., Communications of the ACM, 2021 [[url]](https://cacm.acm.org/magazines/2021/4/251365-the-impossibility-of-fairness/fulltext)

**COMPAS case study**

* ProPublica https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing 
* https://allendowney.github.io/RecidivismCaseStudy/

**Demo's**

* https://research.google.com/bigpicture/attacking-discrimination-in-ml

### Fairness in classification: Individuals

* Fairness through awareness, Dwrok et al., ITCS ’12 [[url]](https://dl.acm.org/doi/10.1145/2090236.2090255)

### Fairness in representations

* Evaluating Gender Bias in Machine Translation, Stanovsky et al., 2019  [[url]](https://www.aclweb.org/anthology/P19-1164/)
* Semantics derived automatically from language corpora contain human-like biases, Caliskan et al. Science 2017 [[url](https://science.sciencemag.org/content/356/6334/183) 
* Google response gender bias in Google Translate https://www.blog.google/products/translate/reducing-gender-bias-google-translate/
* Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings, Bolukbasi et al. NeurIPS 2016 [[url]](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings)
* ConceptNet Numberbatch 17.04: better, less-stereotyped word vectors, Robyn Speer, 2017 [[url]](http://blog.conceptnet.io/posts/2017/conceptnet-numberbatch-17-04-better-less-stereotyped-word-vectors/)


## Interventions

### Pre-processing

* Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods, Zhao et al. NAACL 2018 [[url]](https://www.aclweb.org/anthology/N18-2003.pdf)
* Data preprocessing techniques forclassification without discrimination,Kamiran and Calders, Knowl Inf Syst 2012 [[url]](https://link.springer.com/article/10.1007/s10115-011-0463-8)


### Post-processing

* Equality of Opportunity in SupervisedLearning, Hardt et al., NIPS 2016 [[url]](https://arxiv.org/abs/1610.02413)
* Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search,Geyik et al., KDD 2019 [[url]](https://dl.acm.org/doi/10.1145/3292500.3330691)
* Man is to computer programmer as woman is to homemaker? Debiasing word embeddings, Bolukbasi et al.NIPS 2016 [[url]](https://papers.nips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)


### In-processing

* Fairness-Aware Classifier with Prejudice Remover Regularizer, Kamishima et al. ECML PKDD 2012 [[url]](https://link.springer.com/chapter/10.1007/978-3-642-33486-3_3)
* Mitigating Unwanted Biases with Adversarial Learning, Zhang et al. AIES ’18 [[url]](https://dl.acm.org/doi/10.1145/3278721.3278779)


## Critiques/limitations/broader perspective

* Fairness and Abstraction in Sociotechnical Systems, Selbst et al., FAT* 2019 [[url]](https://doi.org/10.1145/3287560.3287598)
* Human Perceptions of Fairnessin Algorithmic Decision Making: A Case Study of Criminal Risk Prediction, Grgic-Hlaca et al.,WWW 2018 [[url]](https://doi.org/10.1145/3178876.3186138)
* Algorithm Aversion: People Erroneously Avoid Algorithms after Seeing Them Err, Dietvorst et al., Journal of Experimental Psychology 2015 [[url]](https://doi.org/10.1037/xge0000033)
* A systematic review of algorithm aversion in augmented decision making, Burton et al., Journal of BehavioralDecision Making 2020 [[url]](https://doi.org/10.1002/bdm.2155)


## Candidate papers for group presentations

* 50 Years of Test (Un)fairness: Lessons for Machine Learning, Hutchinson and Mitchell, FAT* 2019 [[url]](https://dl.acm.org/doi/10.1145/3287560.3287600)

* Fairness Is Not Static: Deeper Understanding of Long Term Fairness via Simulation Studies, D'Amour et al., FAT* 2020 [[url]](https://dl.acm.org/doi/abs/10.1145/3351095.3372878)

* Beyond Distributive Fairness in Algorithmic Decision Making: Feature Selection for Procedurally Fair Learning, Grgić-Hlača et al., AAAI 2018 [[url]](https://ojs.aaai.org/index.php/AAAI/article/view/11296)

* Fairness Constraints: Mechanisms for Fair Classification, Zafar et al. 2017, AISTATS 2017 [[url]](http://proceedings.mlr.press/v54/zafar17a.html)

* Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints, Zhao et al., EMNLP 2017 [[url]](https://www.aclweb.org/anthology/D17-1323/)

* Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings, Bolukbasi et al., NIPS 2016 [[url]](https://dl.acm.org/doi/10.5555/3157382.3157584) 

* Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification, Buolamwini and Gebru, FAT 2018 [[url]](http://proceedings.mlr.press/v81/buolamwini18a.html)

* Image Representations Learned With Unsupervised Pre-Training Contain Human-like Biases, Steed and Caliskan, FAccT 2021 [[url](https://dl.acm.org/doi/10.1145/3442188.3445932)

* Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search, Geyik et al., KDD 2019 [[url]](https://dl.acm.org/doi/10.1145/3292500.3330691)

* Factors Influencing Perceived Fairness in Algorithmic Decision-Making: Algorithm Outcomes, Development Procedures, and Individual Differences, Wang et al., CHI 2020 [[url]](https://dl.acm.org/doi/10.1145/3313831.3376813)

* ‘It’s Reducing a Human Being to a Percentage’; Perceptions of Justice in Algorithmic Decisions, Binns et al., CHI 2018, [[url]](https://doi.org/10.1145/3173574.3173951) 

* Mitigating Biases in Multimodal Personality Assessment, Shen Yan et al.,  ICMI 2020 [[url]](https://doi.org/10.1145/3382507.3418889)


## Demo's /tools
* https://github.com/google/ml-fairness-gym
* https://fairlearn.github.io

## User experiments
* https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=3503603

## Critiques/limitations on current fairness work 
* Non-portability of Algorithmic Fairness in India https://arxiv.org/pdf/2012.03659.pdf
* Where fairness fails: data, algorithms, and the limits of antidiscrimination discourse https://www.tandfonline.com/doi/abs/10.1080/1369118X.2019.1573912?journalCode=rics20
* Improving Fairness in Machine Learning Systems: What Do Industry Practitioners Need? https://dl.acm.org/doi/abs/10.1145/3290605.3300830



## NLP papers ##

### Embeddings

* Word Embeddings Quantify 100 Years of Gender and Ethnic Stereotypes, Garg et al. PNAS 2017 https://arxiv.org/pdf/1711.08412.pdf
* Learning Gender-Neutral Word Embeddings, Zhao et al. EMNLP 2018 https://arxiv.org/pdf/1809.01496
* Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them, Gonen and Goldberg. NAACL-HLT 2019 https://arxiv.org/pdf/1809.01496
* Understanding the Origins of Bias in Word Embeddings, Brunet et al. ICML 2019, https://arxiv.org/pdf/1810.03611.pdf
* Assessing Social and Intersectional Biases in Contextualized Word Representations, Tan and Celis, NeurIPS 2019. https://papers.nips.cc/paper/9479-assessing-social-and-intersectional-biases-in-contextualized-word-representations.pdf

### Other


* Co-reference resolution: Gender Bias in Coreference Resolution, Rudinger et al. NAACL 2018 https://www.aclweb.org/anthology/N18-2002/  and Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods Zhao et al. NAACL 2018 https://www.aclweb.org/anthology/N18-2003/ 
* Language (Technology) is Power: A Critical Survey of "Bias" in NLP, Blodgett et al. ACL 2020 https://www.aclweb.org/anthology/2020.acl-main.485.pdf 
* What’s in a Name? Reducing Bias in Bios without Access to Protected Attributes, Romanov et al. NAACL-HLT 2019, https://arxiv.org/pdf/1904.05233.pdf
* Mitigating Gender Bias in Natural Language Processing: Literature Review, Sun et al. ACL 2019, https://www.aclweb.org/anthology/P19-1159/
* Reducing Gender Bias in Abusive Language Detection, Park et al. EMNLP 2018, https://www.aclweb.org/anthology/D18-1302
* Fair Bayesian Optimization, Perrone et al., 2020 ICML Workshop on Automated Machine Learning, https://assets.amazon.science/f6/1c/bc38ad454d029ba747529ca03ee6/fair-bayesian-optimization.pdf
* [todo]

## Computer vision

* Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification, Buolamwini and Gebru, Proceedings of Machine Learning Research, http://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf 2018

## Advertisements
* Discrimination in Online Ad Delivery, Sweeney, ACM Queue 2013 https://queue.acm.org/detail.cfm?id=2460278


# Interpretability / explainability

## Books

* Interpretable machine learning, Molnar https://christophm.github.io/interpretable-ml-book/ 

## General readings

* The Mythos of Model Interpretability, Lipton 2018 https://queue.acm.org/detail.cfm?id=3241340
* Guidotti et al. A Survey of Methods for Explaining Black Box Models, ACM Computing Surveys (CSUR) 2019 https://dl.acm.org/citation.cfm?id=3236009
* Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead, Rudin, Nature Machine Intelligence 2019 https://www.nature.com/articles/s42256-019-0048-x
* Towards A Rigorous Science of Interpretable Machine Learning,  Doshi-Velez and Kim 2017, https://arxiv.org/abs/1702.08608
* Explanation in Artificial Intelligence: Insights from the Social Sciences, Miller, 2019, https://arxiv.org/abs/1706.07269

## Counterfactuals/contrastsets
* Learning the Difference that Makes a Difference with Counterfactually-Augmented Data https://arxiv.org/abs/1909.12434
* Evaluating NLP Models via Contrast Sets https://arxiv.org/pdf/2004.02709.pdf


## Local explanations

* “Why Should I Trust You?” Explaining the Predictions of Any Classifier, 
Ribeiro et al. KDD 2016 https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf
* A Unified Approach to Interpreting Model Predictions, Lundberg and Lee, NeurIPS 2017 http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf
* From local explanations to global understanding with explainable AI for trees. Lundberg, S.M. et al., Nature Machine Intelligence 2, 56–67 (2020). https://doi-org.proxy.library.uu.nl/10.1038/s42256-019-0138-9 , available from https://arxiv.org/abs/1905.04610 
* Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods, Slack et al., AIES'20 https://arxiv.org/abs/1911.02508
* A Textual Adversial Attack Paper collection https://github.com/thunlp/TAADpapers
* Understanding Black-box Predictions via Influence Functions, Koh and Liang, ICML 2017 http://proceedings.mlr.press/v70/koh17a/koh17a.pdf
* Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications, Pezeshkpour et al. NAACL-HLT 2019 https://arxiv.org/pdf/1905.00563
* Layer-Wise Relevance Propagation: An Overview, Montavon et al. Explainable AI: Interpreting, Explaining and Visualizing Deep Learning  2019
https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10
* Beyond saliency: understanding convolutional neural networks from saliency prediction on layer-wise relevance propagation, Heyi Li, Yunke Tian, Klaus Mueller, and Xin Chen. Image and Vision Computing 83 (2019). https://www.sciencedirect.com/science/article/pii/S0262885619300149
* Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR, Wachter et al. 2018 https://arxiv.org/abs/1711.00399
* Gradcam: Visual explanations from deep networks via gradient-based localization, Selvaraju et al. ICCV 2017. https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf
* [Todo] Gradient based methods

## Explainability and Bias in Multimodal Affective Computing 
* "Modeling, Recognizing, and Explaining Apparent Personality from Videos."  Escalante et al., IEEE Transactions on Affective Computing (2020), https://ieeexplore.ieee.org/abstract/document/8999746 also available from https://arxiv.org/abs/1802.00745
* Investigating Bias and Fairness in Facial Expression Recognition, Xu et al., ECCV'20 Workshop on Fair Face Recognition and Analysis. https://arxiv.org/abs/2007.10075

## Evaluation

* Evaluating the visualization of what a deep neural network has learned, Samek et al.  IEEE Transactions on Neural Networks and Learning Systems 2017 https://ieeexplore.ieee.org/document/7552539 
* Towards Faithfully Interpretable NLP Systems:How Should We Define and Evaluate Faithfulness? https://www.aclweb.org/anthology/2020.acl-main.386.pdf
* [Todo] User experiments, trust..

## Datasets

* https://www.excavating.ai/
* Possible assignment: Let students read and analyze the datasets themselves.

## Visualization

* [todo]


## Toolkits
* AllenNLP Interpret  https://allennlp.org/interpret
* [todo]
* Captum https://captum.ai/
* Responsible AI Tensorflow https://www.tensorflow.org/resources/responsible-ai

# Related courses
* Interpretability and Explainability in Machine Learning, COMPSCI 282BR, Harvard University: https://interpretable-ml-class.github.io/ (2019)
* CS 294: Fairness in Machine Learning, UC Berkeley (2017): https://fairmlclass.github.io/
* Trustworthy Machine Learning, University of Toronto: https://www.papernot.fr/teaching/f19-trustworthy-ml (2019)
* Fairness, Explainability, and Accountability for ML, ETH Zurich: https://las.inf.ethz.ch/teaching/feaml-s19 (2019)
* Human-Centered Machine Learning, Saarland University: http://courses.mpi-sws.org/hcml-ws18/ (2018)
* Human-centered Machine Learning, University of Colorado Boulder. https://chenhaot.com/courses/hcml/home.html (2018)
* LING 575 — Ethics in NLP: Including Society in Discourse & Design, University of Washington https://ryan.georgi.cc/courses/575-ethics-win-19/ (2019)
* Computational Ethics for NLP, CMU: http://demo.clab.cs.cmu.edu/ethical_nlp/ (2019)
* https://github.com/stanford-policylab/law-order-algo
* Seminar on Ethical and Social Issues in Natural Language Processing  (Stanford 2020) https://docs.google.com/document/d/1zujyrSTiQ-HKQ66oPDiswihe55jUY7pwHpQPwMHTBFI/edit#

# Other
* https://pair.withgoogle.com/


# Tutorials
* Bias and Fairness in Natural Language Processing EMNLP 2019 http://web.cs.ucla.edu/~kwchang/talks/emnlp19-fairnlp/ 
