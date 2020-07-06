
# Fairness

## Books ##

* Online book: Fairness and machine learning, Limitations and Opportunities, Solon Barocas, Moritz Hardt, Arvind Narayanan https://fairmlbook.org/

## COMPAS case study ##

* ProPublica https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing 
* [Todo]

## Demo's /tools ##
* https://github.com/google/ml-fairness-gym

## User experiments ##
* https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=3503603

## NLP papers ##

### Embeddings
* Semantics derived automatically from language corpora contain human-like biases, Caliskan et al. Science 2017 https://science.sciencemag.org/content/356/6334/183
* Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings, Bolukbasi et al. NeurIPS 2016 https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings
* Word Embeddings Quantify 100 Years of Gender and Ethnic Stereotypes, Garg et al. PNAS 2017 https://arxiv.org/pdf/1711.08412.pdf
* Learning Gender-Neutral Word Embeddings, Zhao et al. EMNLP 2018 https://arxiv.org/pdf/1809.01496
* Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them, Gonen and Goldberg. NAACL-HLT 2019 https://arxiv.org/pdf/1809.01496
* Understanding the Origins of Bias in Word Embeddings, Brunet et al. ICML 2019, https://arxiv.org/pdf/1810.03611.pdf
* Assessing Social and Intersectional Biases in Contextualized Word Representations, Tan and Celis, NeurIPS 2019. https://papers.nips.cc/paper/9479-assessing-social-and-intersectional-biases-in-contextualized-word-representations.pdf

### Other
* The Risk of Racial Bias in Hate Speech Detection, Sap et al., ACL 2019 https://www.aclweb.org/anthology/P19-1163/
* Google response gender bias in Google Translate https://www.blog.google/products/translate/reducing-gender-bias-google-translate/
* Co-reference resolution: Gender Bias in Coreference Resolution, Rudinger et al. NAACL 2018 https://www.aclweb.org/anthology/N18-2002/  and Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods Zhao et al. NAACL 2018 https://www.aclweb.org/anthology/N18-2003/ 
* Language (Technology) is Power: A Critical Survey of "Bias" in NLP, Blodgett et al. ACL 2020 https://www.aclweb.org/anthology/2020.acl-main.485.pdf 
* What’s in a Name? Reducing Bias in Bios without Access to Protected Attributes, Romanov et al. NAACL-HLT 2019, https://arxiv.org/pdf/1904.05233.pdf
* Mitigating Gender Bias in Natural Language Processing: Literature Review, Sun et al. ACL 2019, https://www.aclweb.org/anthology/P19-1159/
* Reducing Gender Bias in Abusive Language Detection, Park et al. EMNLP 2018, https://www.aclweb.org/anthology/D18-1302
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
* Escalante, Hugo Jair, et al. "Modeling, Recognizing, and Explaining Apparent Personality from Videos." IEEE Transactions on Affective Computing (2020), https://ieeexplore.ieee.org/abstract/document/8999746 also available from https://arxiv.org/abs/1802.00745

## Evaluation

* Evaluating the visualization of what a deep neural network has learned, Samek et al.  IEEE Transactions on Neural Networks and Learning Systems 2017 https://ieeexplore.ieee.org/document/7552539 
* Towards Faithfully Interpretable NLP Systems:How Should We Define and Evaluate Faithfulness? https://www.aclweb.org/anthology/2020.acl-main.386.pdf
* [Todo] User experiments, trust..

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
