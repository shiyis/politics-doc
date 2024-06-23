---
title: Measuring Ideal Points
layout: home
nav_order: 2
---

{% include latex_template.html %}


<!--
{: .fs-5 .fw-700 .text-delta } -->
<div class="text-center">


Measuring Political Subjectivity with Variational Auto Encoding Methods
</div>
{: .fs-5 .fw-700 .text-delta}


The model performs inference using [variational inference](https://arxiv.org/abs/1601.00670) with [reparameterization gradients](https://arxiv.org/abs/1312.6114) [](href=https://arxiv.org/abs/1401.4082). A **VAE** or Variational Auto Encoding technique would learn to represent the underlying patterns and structures of cat pictures in a lower-dimensional latent space.

---

What this means is imagine you have a bunch of cat pictures, and you want to generate new, realistic cat pictures that you've never seen before. **This latent space would capture the essential features of cat pictures, such as the shape of the ears, the color of the fur, and the length of the tail.** The **VAE** would then use this latent space to generate new cat pictures that are similar to the ones it's seen before.,

In the context of topic modeling, **VAEs** can be used to learn a probabilistic representation of topics in a document collection. The latent space would capture the underlying themes and patterns in the documents, and the **VAE** would generate topic distributions for each document based on this latent space.


This is an extension of another popular algorithm the [Latent Dirichlet Allocation (LDA)](). In the context of textual topic modeling, variational inference helps approximate the **posterior distribution of latent variables, such as the distribution of topics in documents and words in topics.**

In variational inference, we need to specify a family of distributions from which we will choose an approximation to the true (but often intractable) posterior distribution. This family of distributions is called the variational family. **Common choices for the variational family include mean-field variational families.**

Now we want to see above descriptions materialized in actual formulas.

The variational family is set to be the mean-field family, meaning the latent variables factorize over documents $d$, topics $k$, and authors $$s$$:


  $$q_\phi(\theta, \beta, \eta, x) = \prod_{d,k,s} q(\theta_d)q(\beta_k)q(\eta_k)q(x_s)$$


Lognormal factors are used for the positive variables and Gaussian factors for the real variables:

 $$q(\theta_d) = \text{LogNormal}_K(\mu_{\theta_d}\sigma^2_{\theta_d})\\$$

 $$q(\beta_k) = \text{LogNormal}_V(\mu_{\beta_k}, \sigma^2_{\beta_k})\\$$

 $$q(\eta_k) = \mathcal{N}_V(\mu_{\eta_k}, \sigma^2_{\eta_k})\\$$

 $$q(x_s) = \mathcal{N}(\mu_{x_s}, \sigma^2_{x_s})\\$$


Again, because it is intractable to evaluate the posterior distribution $p(\theta, \beta, \eta, x  \|  y)$, so the posterior is estimated with a distribution $q_\phi(\theta, \beta,\eta,x)$, parameterized by $\phi$ through minimizing the KL-Divergence between $q$ and the posterior (put simple is the distance between these two distributions), which is equivalent to maximizing the ELBO (or the Evidence Lower Bound):


  $$\mathbb{L}_{\theta,\phi}(\mathbf{x})=\mathbb{E}_{q_{\phi}(\mathbf{z} \| \mathbf{x})}[\log p_{\theta}(\mathbf{x},\mathbf{z})-\log q_{\phi}(\mathbf{z} \| \mathbf{x})]$$






Thus, the goal is to maximize the ELBO with respect to $$\phi = \{\mu_\theta, \sigma_\theta, \mu_\beta, \sigma_\beta,\mu_\eta, \sigma_\eta, \mu_x, \sigma_x\}$$.,
The most important is the initializations of the variational parameters $$\phi$$ and their respective variational distributions:

$$\textbf{loc}\text{: location variables } \mu$$

$$\textbf{scale}\text{: scale variables } \sigma$$

$$\mu_\eta \text{: ideological_topic_loc} $$

$$\sigma_\eta \text{: ideological_topic_scale}$$


The corresponding variational distribution is `ideological_topic_distribution`.  Below summarizes the above formulas in plainer language.

{: .note}
**In mean-field variational inference, it is assumed that the posterior distribution factorizes across latent variables.** This means that each latent variable is independent of the others given parameters. For LDA, these parameters might include the distribution of topics in documents and words in topics.

{: .note}
**Start with some initial parameters for the variational family. Optimize these parameters to make the approximating distribution as close as possible to the true posterior distribution.** In this case, it's through maximizing the evidence lower bound or minimizing the KL-Divergence between these two probability distributions.

{: .note}
**The parameters of the variational family can be interpreted as the estimated distributions of topics.** These distributions are used to project latent information onto the documents and words. Each document gets a distribution over topics, and each topic gets a distribution over words.

{: .note}
**The algorithm, through this variational inference process, discovers latent topics in the corpora based on how words co-occur across documents.** Alternatively, it's measuring the pointwise mutual information between two probability distributions.

The default corpus for this Colab notebook is, [Senate speeches](https://data.stanford.edu/congress_text). The project also used the following corpora:


Tweets from 2022 Democratic presidential candidates. To replicate the whole process with my own Twitter data, I followed the steps below

  * `counts.npz`: a `[num_documents, num_words]` [sparse CSR matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html) containing the word counts for each document.
  * `author_indices.npy`: a `[num_documents]` vector where each entry is an integer in the set `{0, 1, ..., num_authors - 1}`, indicating the author of the corresponding document in `counts.npz`.
  * `vocabulary.txt`: a `[num_words]` - length file where each line denotes the corresponding word in the vocabulary.
  * `author_map.txt`: a `[num_authors]` - length file where each line denotes the name of an author in the corpus.

Please checkout this [notebook](https://colab.research.google.com/github/pyro-ppl/numpyro/blob/5291d0627d68598cf78b8ea97c540268660925c1/notebooks/source/tbip.ipynb) for the full implementation in Python.
