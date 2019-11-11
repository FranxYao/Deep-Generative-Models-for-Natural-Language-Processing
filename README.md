# Deep-Generative-Models-for-Natural-Language-Processing
DGMs 4 NLP, Deep Generative Models for Natural Language Processing, resources, conference mapping and paper list

Yao Fu, Columbia University, yao.fu@columbia.edu

![title](https://github.com/Francix/Deep-Generative-Models-for-Natural-Language-Processing/blob/master/src/titlepage.jpeg)

----

When talking about deep generative models, one usually refers to three model families: the Variational Autoencoders (VAEs), the Generative Adversarial Networks (GANs), and the Normalizing Flows.

Amoung the three model families, we will focus more on VAEs side since they are more effective. Whether GAN really works is still an open question. The effectiveness of GANs is more like the discriminator's regularization, rather than the 'generative' part. Or correct me if I am wrong.

Many discrete structures are involved in VAE models for NLP. Inference over these structures is tricky and smart. Many of them deserve to know.

## Resources 

Before the beginning of our journey, the fundation of the DGMs is build upon probabilistic graphical models. So we start from these models 

#### Blei's Foundation of Graphical Models course, STAT 6701 at Columbia ([link](http://www.cs.columbia.edu/~blei/fogm/2019F/index.html))
* This course talks about the foudations of probabilistic modeling, graphical models, and approximate inference. 

#### Xing's Probabilistic Graphical Models, 10-708 at CMU ([link](https://sailinglab.github.io/pgm-spring-2019/))
* This is a really heavy course with extensive materials. There are 5 modules in total: exact inference, approximate inference, DGMs, reinforcement learning, and non-parameterics. All lecture notes, vedio recordings, and homeworks are open-sourced. 

#### Collins' Natural Language Processing, COMS 4995 at Columbia ([link](http://www.cs.columbia.edu/~mcollins/cs4705-spring2019/))
* This course may look like an NLP course, but it has a graphical models core (with an NLP surface.) Many structural inference methods are introduced. Also take a look at many related notes from [Collins' homepage](http://www.cs.columbia.edu/~mcollins/)

Also there are many books that worth learning.

#### Pattern Recognition and Machine Learning. Christopher M. Bishop. 2006
* The _core part_, according to my own understanding, of this book, should be section 8 - 13, especially section 10 since this is the section that introduces variational inference. 
* If you only have time reading one chapter, read section 10. 
* This book is also a great book for building systemacal knowledge of graphical models. 

#### Machine Learning: A Probabilistic Perspective. Kevin P. Murphy. 2012
* Compared with the PRML Bishop book, this book may be used as a super-detailed handbook for various graphical models and inference methods, rather than a textbook, because it is super-detailed. 
* Basically you can find a galary of every classical graphical models from this book. 

Now we go to the realm of DGMs. 

#### Wilker Aziz's DGM Landscape ([link](http://wilkeraziz.github.io/pages/landscape))
* This is a great guidebook for VI. It is a graph over the VI literature and discuss the connections of different techniques. Definitely go over this to have a rough sense/ go deep about DGMs 

#### A Tutorial on Deep Latent Variable Models of Natural Language ([link](https://arxiv.org/abs/1812.06834)), EMNLP 18 
* Yoon Kim, Sam Wiseman and Alexander M. Rush, Havard

#### Deep Generative Models for Natural Language Processing, Ph.D. Thesis 17, ([link](https://ora.ox.ac.uk/catalog/uuid:e4e1f1f9-e507-4754-a0ab-0246f1e1e258/download_file?file_format=pdf&safe_filename=PhD_Thesis_of_University_of_Oxford%2B%25287%2529.pdf&type_of_work=Thesis))
* Yishu Miao, Oxford

#### Columbia STAT 8201, Deep Generative Models ([link](http://stat.columbia.edu/~cunningham/teaching/GR8201/))
* This is the seminar course I took at Columbia. The first part of this course focus on VAEs and the second part focus on GANs. 

#### Stanford CS 236, Deep Generative Models ([link](https://deepgenerativemodels.github.io/))

#### NYU Deep Generative Models ([link](https://cs.nyu.edu/courses/spring18/CSCI-GA.3033-022/))

#### U Toronto [CS 2541](https://www.cs.toronto.edu/~duvenaud/courses/csc2541/index.html) Differentiable Inference and Generative Models, [CS 2547](https://duvenaud.github.io/learn-discrete/) Learning Discrete Latent Structures.  

----

The order of the papers are not very well organized. I will improve it. 

## NLP Side 

We will focus on two topics: generation and structural inference. We start from generation

### VAEs

#### Generating Sentences from a Continuous Space, CoNLL 15
* Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz, Samy Bengio
* This seems to be the first paper using VAEs for NLP.
* **BUT** it seems that many of the results in the paper are not that solid/ could be improved by better models ("not that solid" seems to be not a proper word choice, I am not a native speaker so appologize for (perhaps) the improper wording, please give me suggessions on how to critisize with suitable words)
* An important point of this paper is about the posterior collapse. This problem is addressed by the following papers.

#### Neural variational inference for text processing, ICML 16 
* Yishu Miao, Lei Yu, Phil Blunsom, Deepmind

#### Improved Variational Autoencoders for Text Modeling using Dilated Convolutions, ICML 17 
* Zichao Yang, Zhiting Hu, Ruslan Salakhutdinov, Taylor Berg-Kirkpatrick

#### Spherical Latent Spaces for Stable Variational Autoencoders, EMNLP 18 
* Jiacheng Xu and Greg Durrett, UT Austin
* A uniform distribution on a unit sphere is helpful to the posterior problem. 

#### Adversarially Regularized Autoencoders, ICML 18 
* Jake (Junbo) Zhao, Yoon Kim, Kelly Zhang, Alexander M. Rush, Yann LeCun. NYU, Havard, FAIR
* A wrapup of the major VAE/ GANs 
* A learned prior to tackle the posterior collapse. 
* Although this paper looks like more ML, but essentially it tackles an NLP problem. I [presented this paper](src/annotated_arae.pdf) in the Columbia DGM seminar course. 

#### Semi-amortized variational autoencoders, ICML 18 
* Yoon Kim, Sam Wiseman, Andrew C. Miller, David Sontag, Alexander M. Rush, Havard
* The **posterior collapse** phenomenon: the variational posterior collapses to the prior and the generative model ignores the latent variable (Dispite all the other stuffs in the intro, I think this is the most important point/ motivation of this paper since the whole NLP community suffer from this for a long time). 
* SVI: view the variational posterior as a model parameter, optimize over is (i.e. the posterior dist. parameter)
* AVI: view the variational posterior as a output of the recognition network (rather than the model parameter), Optimize the recognition network. 
* Semi-armortized VAE: first use a recognition network to predict the variational parameter (the armortized part), then optimize over this parameter (stochastic part.)
* The implementation heavily involves optimization techniques/ tricks. 
* Experiments: higher KL (indicating that latent variables are not collepsed) and lower ppl (performance metrics). 
* Saliency analysis: a visualization of the relationship between the latent variable and the input/ output, as an example of interpretability (or just random guess and coincidence, who knows). 

#### Lagging Inference Networks and Posterior Collapse in Variational Autoencoders, ICLR 19 
* Junxian He, Daniel Spokoyny, Graham Neubig, Taylor Berg-Kirkpatrick

#### Avoiding Latent Variable Collapse with Generative Skip Models, AISTATS 19 
* Adji B. Dieng, Yoon Kim, Alexander M. Rush, David M. Blei

### Structural Inference

Now we talk about structural inference. This induces chunking, tagging and parsing. 

#### An introduction to Conditional Random Fields. 

#### Structured Attention Networks. 

#### Differentiable Dynamic Programming for Structured Prediction and Attention 

#### Inside-Outside and Forward-Backward Algorithms Are Just Backprop

#### Recurrent Neural Network Grammars. NAACL 16
* Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah Smith.
* A transaction based generative model to model the joint prob of trees and sentences. 
* Smart inference trick: use importance sampling to calculate the sentence marginal prob. Use a discriminative model as the proposal dist. 

Later the RNNG model is extended to be an unsupervised version:

#### Unsupervised Recurrent Neural Network Grammars, NAACL 19 
* Yoon Kin, Alexander Rush, Lei Yu, Adhiguna Kuncoro, Chris Dyer, and Gabor Melis
* Compared with the above perturb-and-parse paper, this paper does not use continuous relexation of the sampling over the CRF, so it use the score function estimator with control variate. 

#### Differentiable Perturb-and-Parse: Semi-Supervised Parsing with a Structured Variational Autoencoder, ICLR 19
* Caio Corro, Ivan Titov, Edinburgh
* Reparameterize the sampling from a CRF by using gumbel perturbation (so one can inject randomness to the potential) and continuous relexation of Eisner (so one can perform efficient inference). Smart move! 


### The Gumbel Trick

#### Categorical Reparameterization with Gumbel-Softmax

#### The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables.

#### Reparameterizable Subset Sampling via Continuous Relaxations

#### Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement. ICML 19
* Wouter Kool, Herke van Hoof, Max Welling
* Gumbel topk, stochastic differentiable beam search 

----

## ML Side 

Now the ML side, we start from VAEs

#### Auto-Encoding Variational Bayes, Arxiv 13 
* Diederik P. Kingma, Max Welling

#### Variational Inference: A Review for Statisticians, Arxiv 18
* David M. Blei, Alp Kucukelbir, Jon D. McAuliffe 

More on reparameterization: 

#### Stochastic Backpropagation through Mixture Density Distributions, Arxiv 16
* Alex Graves
* This paper gives a method for reparameterize Gaussian Mixture 


Now GANs

#### Generative Adversarial Networks, NIPS 14
* Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
* GAN origin 
* This original GAN paper use the KL divergence to measure the distance between probability distributions, which may lead to the vanishing of gradient. To tackle this problem, the wassertein GAN is proposed with the earch mover distance. The following two papers shows the birth of wGAN.

#### Towards principled methods for training generative adversarial networks, ICLR 2017 
* Martin Arjovsky and Leon Bottou
* Discusses the distance between distributions, but uses many hacky methods.

#### Wasserstein GAN
* Martin Arjovsky, Soumith Chintala, Léon Bottou
* The principled methods, born from hacky methods. 


Then we look at normalizing flows: 

#### Variational Inference with Normalizing Flows, ICML 15 
* Danilo Jimenez Rezende, Shakir Mohamed

#### Improved Variational Inference with Inverse Autoregressive Flow
* Diederik P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling

#### Learning About Language with Normalizing Flows 
* Graham Neubig, CMU, [slides](http://www.phontron.com/slides/neubig19generative.pdf)

----

## Paraphrase and Language Diversity 

#### A Deep Generative Framework for Paraphrase Generation, AAAI 18
* Ankush Gupta, Arvind Agarwal, Prawaan Singh, Piyush Rai 

#### Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization, NIPS 18
* Yizhe Zhang, Michel Galley, Jianfeng Gao, Zhe Gan, Xiujun Li, Chris Brockett, Bill Dolan


----

## Topic-aware Langauge Generation

#### Discovering Discrete Latent Topics with Neural Variational Inference, ICML 17 
* Yishu Miao, Edward Grefenstette, Phil Blunsom. Oxford

#### Topic-Guided Variational Autoencoders for Text Generation, NAACL 19 
* Wenlin Wang, Zhe Gan, Hongteng Xu, Ruiyi Zhang, Guoyin Wang, Dinghan Shen, Changyou Chen, Lawrence Carin. Duke & MS & Infinia & U Buffalo
* A neural topic model 
* A Gaussian Mixture latent prior and posterior 
* A Householder Flow for inferring the Gaussian Mixture

#### TopicRNN: A Recurrent Neural Network with Long-Range Semantic Dependency, ICLR 17 
* Adji B. Dieng, Chong Wang, Jianfeng Gao, John William Paisley

#### Topic Compositional Neural Language Model, AISTATS 18 
* Wenlin Wang, Zhe Gan, Wenqi Wang, Dinghan Shen, Jiaji Huang, Wei Ping, Sanjeev Satheesh, Lawrence Carin

#### Topic Aware Neural Response Generation, AAAI 17 
* Chen Xing, Wei Wu, Yu Wu, Jie Liu, Yalou Huang, Ming Zhou, Wei-Ying Ma
