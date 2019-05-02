# Deep-Generative-Models-for-Natural-Language-Processing
Deep Generative Models for Natural Language Processing, resources, conference mapping and paper list

![title](https://github.com/Francix/Deep-Generative-Models-for-Natural-Language-Processing/blob/master/src/titlepage.jpeg)

----

## Resources 

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

----

## NLP Side 

Actually, whenever talking about using the DGM techniques for NLP, I always feel these are like black magic compared with seq2seq and MLE. 

#### Differentiable Perturb-and-Parse: Semi-Supervised Parsing with a Structured Variational Autoencoder
* Caio Corro, Ivan Titov, Edinburgh
* Model parsing tree as MRF, differentiable sampling by perturbing. 

#### Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement. Arxiv 19
* Wouter Kool, Herke van Hoof, Max Welling
* Gumbel topk, stochastic differentiable beam search 

#### Spherical Latent Spaces for Stable Variational Autoencoders, EMNLP 18 
* Jiacheng Xu and Greg Durrett, UT Austin

#### Semi-amortized variational autoencoders, ICML 18 
* Yoon Kim, Sam Wiseman, Andrew C. Miller, David Sontag, Alexander M. Rush, Havard
* The **posterior collapse** phenomenon: the variational posterior collapses to the prior and the generative model ignores the latent variable (Dispite all the other stuffs in the intro, I think this is the most important point/ motivation of this paper since the whole NLP community suffer from this for a long time). 
* SVI: view the variational posterior as a model parameter, optimize over is (i.e. the posterior dist. parameter)
* AVI: view the variational posterior as a output of the recognition network (rather than the model parameter), Optimize the recognition network. 
* Semi-armortized VAE: first use a recognition network to predict the variational parameter (the armortized part), then optimize over this parameter (stochastic part.)
* The implementation heavily involves optimization techniques/ tricks. 
* Experiments: higher KL (indicating that latent variables are not collepsed) and lower ppl (performance metrics). 
* Saliency analysis: a visualization of the relationship between the latent variable and the input/ output, as an example of interpretability (or just random guess and coincidence, who knows). 

#### Neural variational inference for text processing, ICML 16 
* Yishu Miao, Lei Yu, Phil Blunsom, Deepmind

#### Lagging Inference Networks and Posterior Collapse in Variational Autoencoders, ICLR 19 
* Junxian He, Daniel Spokoyny, Graham Neubig, Taylor Berg-Kirkpatrick

#### Avoiding Latent Variable Collapse with Generative Skip Models, AISTATS 19 
* Adji B. Dieng, Yoon Kim, Alexander M. Rush, David M. Blei

#### Improved Variational Autoencoders for Text Modeling using Dilated Convolutions, ICML 17 
* Zichao Yang, Zhiting Hu, Ruslan Salakhutdinov, Taylor Berg-Kirkpatrick

#### Discovering Discrete Latent Topics with Neural Variational Inference, ICML 17 
* Yishu Miao, Edward Grefenstette, Phil Blunsom. Oxford

----

## ML Side 

#### Adversarially Regularized Autoencoders, ICML 18 
* Jake (Junbo) Zhao, Yoon Kim, Kelly Zhang, Alexander M. Rush, Yann LeCun. NYU, Havard, FAIR
* A wrapup of the major VAE/ GANs 
* Although this paper looks like more ML, but essentially it tackles an NLP problem. I [presented this paper](src/annotated_arae.pdf) in the Columbia DGM seminar course. 

#### Stochastic Backpropagation through Mixture Density Distributions, Arxiv 16
* Alex Graves
* This paper gives a method for reparameterize Gaussian Mixture 

#### Auto-Encoding Variational Bayes, Arxiv 13 
* Diederik P. Kingma, Max Welling

#### Variational Inference: A Review for Statisticians, Arxiv 18
* David M. Blei, Alp Kucukelbir, Jon D. McAuliffe 

#### Differentiable Subset Sampling
* Sang Michael Xie and Stefano Ermon
* The gumbel-softmax gives sample of a single entry from a set. 
* Instead of sampling one single entry, we want to sample a subset of size k from a set of size n. How to make this procedure differentiable? 
* Inspired by the weighted reservoir sampling, we construct Gumbel-weights for each entry. (Also recall the relationship of Uniform dist., Exponential dist., Gumbel dist., and Discrete dist. we discussed last week.)
* Then we use a differentiable top-k procedure to get a k-hot vector. This procedure repeat softmax k times, after each step, it set the weight of the previously sampled entry to be -inf (softly). I think this procedure is smart. 


----

## Paraphrase and Language Diversity 

#### A Deep Generative Framework for Paraphrase Generation, AAAI 18
* Ankush Gupta, Arvind Agarwal, Prawaan Singh, Piyush Rai 

#### DP-GAN: Diversity-Promoting Generative Adversarial Network for Generating Informative and Diversified Text, EMNLP 18 
* Jingjing Xu, Xuancheng Ren, Junyang Lin, Xu Sun

#### Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization, NIPS 18
* Yizhe Zhang, Michel Galley, Jianfeng Gao, Zhe Gan, Xiujun Li, Chris Brockett, Bill Dolan


----

## Topic-aware Langauge Generation

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
