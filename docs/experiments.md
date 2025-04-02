# Gaia Network Experiment Design

## Introduction

This document outlines the design for experiments to evaluate the Gaia Network prototype. The goal is to provide a clear and structured approach to testing the prototype's functionality and performance, with a focus on the most novel and unproven components.

## Experiment 1: Trust Modeling and Incentives for Quality

### 1.1 Objective

Evaluate the effectiveness of the trust modeling component in the Gaia Network prototype. Our high-level hypotheses are:

1. Trust modeling can help nodes make better decisions by grounding their inferences in reliable data.
2. Incentives can further improve the quality of data provided by nodes.
3. The basic setup is resilient against spam and malicious nodes.

### 1.2 Experimental Setup

#### 1.2.1 Network Setup

The experiment will be run on a simulated network. Nodes can be of two types:

- **Providers**: These nodes provide noisy measurements of observable variables.
- **Customers or Decision-Makers**: These nodes perform inference of latent variables based on the data provided by the providers.

#### 1.2.2 World Model

The experiment runs as a series of rounds, with each round consisting of the following steps:

1. **Data Collection**: Providers collect noisy measurements of observable variables.
2. **Inference**: Customers perform inference of latent variables based on the data provided by the providers.
3. **Reward Calculation**: Providers are rewarded based on the accuracy of their measurements.
4. **Trust Update**: Customers update their trust in providers based on the accuracy of their measurements.

Each customer $i$ has a classification model with a **shared** categorical latent variable $Z$ and a set of categorical observable variables (modalities) $X_{m}$. While the underlying data-generating process follows a true likelihood $P_{true}(X_m = x | Z = z)$, each customer operates based on their **own, potentially incorrect, fixed likelihood model** $L_{i,m,z,x} = P_i(X_m = x | Z = z)$. The customer's posterior belief about the shared latent state $Z$, denoted $P_i(Z | \mathcal{X}_i)$, is updated using their own model via exact inference:

$$P_i(Z=z | \mathcal{X}_i \cup \{x_{m}\}) \propto P_i(X_m = x_m | Z=z) P_i(Z=z | \mathcal{X}_i)$$

Each provider $j$ provides a single observable variable $X_{m_j}$. Typically $\#\{j\} > \#\{m\}$, such that multiple providers can provide the same observable variable. Each customer estimates $\rho_{jm}$, the reliability of provider $j$ for observable variable $m$, and uses it to convert the provider's measurement $X_{m_j} = \delta(x, x_{m_j})$ into a "soft observation" $\tilde{X}_{m_j}$ by mixing the provider's measurement with a uniform prior:

$$P(\tilde{X}_{m_j} = x_{m_j}) = \rho_{jm} \delta(x, x_{m_j}) + (1 - \rho_{jm}) \frac{1}{\#\{x\}}$$

The customer's reward $r_i$ is assumed to be simply the log posterior of the true latent category (as would be the case if, for instance, the customer were a portfolio manager allocating capital across different assets and the payoffs were 1 for the true "good" asset and 0 for the others). Hence we can treat this decision problem as a pure inference problem.

$$r_i = \log P_i(Z = \hat{z} | \mathcal{X}_i)$$

Customers perform active learning to select the providers they will use to update their belief about the latent variable $Z$. Starting with the prior $P(Z | \emptyset)$ (which might be denoted $P_i(Z | \emptyset)$ if priors differ, but let's assume shared prior for now), the customer will iteratively build up its set of observations $\mathcal{X}_i$ by selecting the next provider $j$ to use which maximizes the expected increase in their reward (calculated using their own model $P_i$) due to the provision of a new observation:

$$\arg\max_{j} \mathbb{E}_{x_{m_j} \sim P_i(x_{m_j}|\mathcal{X}_i)} \left[ \text{KL}( P_i(Z | \mathcal{X}_i \cup \{x_{m_j}\}) || P_i(Z | \mathcal{X}_i) ) \right]$$

Because the customer's reward is simply the log posterior of the true latent category, the expression to be maximized is the Expected Information Gain (EIG), which equals the expected KL divergence between the posterior belief after observing the (unknown) outcome $x_{m_j}$ from provider $j$ and the current posterior belief. This can be written as:

*   Let $P_i \in \mathbb{R}^{N_z}$ be customer $i$'s current posterior probability vector $(P_i)_z = P_i(Z=z|\mathcal{X}_i)$.
*   Let $L_{i,m} \in \mathbb{R}^{N_z \times N_x}$ be **customer $i$'s likelihood matrix** for modality $m$, $(L_{i,m})_{zx} = P_i(X_m=x | Z=z)$.
*   Let $\rho_{ij}$ be customer $i$'s reliability estimate for provider $j$ (providing modality $m_j$).
*   Let $U \in \mathbb{R}^{N_x}$ be the uniform distribution vector $(U)_x = 1/N_x$, and $\mathbf{1}_{N_z} \in \mathbb{R}^{N_z}$ be a vector of ones.
*   The reliability-adjusted likelihood matrix (based on customer $i$'s model) is $\tilde{L}_{ij} = \rho_{ij} L_{i, m_j} + (1-\rho_{ij}) \mathbf{1}_{N_z} U^T \in \mathbb{R}^{N_z \times N_x}$.
*   The predictive distribution vector is $p_{ij}^x = \tilde{L}_{ij}^T P_i \in \mathbb{R}^{N_x}$, where $(p_{ij}^x)_x = P_i(X_{m_j}=x|\mathcal{X}_i)$.
*   The posterior belief vector after observing $x$ (using $P_i$'s model) is $P_{i|x} = \frac{P_i \odot \tilde{L}_{ij}[:, x]}{(p_{ij}^x)_x} \in \mathbb{R}^{N_z}$, where $(P_i | x)_z = P_i(Z=z | \mathcal{X}_i \cup \{x\})$.
*   Let $H(V) = - \sum_k V_k \log V_k$ be the Shannon entropy function.

The EIG for selecting provider $j$ is then:

$$EIG_{ij} = H(P_i) - (p_{ij}^x)^T H_{i,Z|X,j}$$

where $H_{i,Z|X,j} \in \mathbb{R}^{N_x}$ is the vector where $(H_{i,Z|X,j})_x = H(P_{i|x})$ represents the entropy of customer $i$'s belief given observation $x$ from provider $j$. The sum is the expected posterior entropy for customer $i$ considering provider $j$, $\mathbb{E}_{x \sim p_{ij}^x}[H(Z|X=x)] = (p_{ij}^x)^T H_{i,Z|X,j}$. 

The customer will iteratively select $j$ such that $\tilde{X}_{m_j} \not\in \mathcal{X}_i$ which maximizes $EIG_{ij}$. This process continues until one of the following conditions is met:

*   $\mathcal{X}_i$ reaches a certain maximum size (corresponding to depleting an information budget).
*   $EIG_{ij} < 0$ (corresponding to the customer's belief that the new observation would not improve their reward).
*   The customer has no more providers to select from.
*   The posterior entropy $H(P_i(Z | \mathcal{X}_i))$ is below a certain threshold (corresponding to the customer's belief that the new observation would not improve their reward).

Once this process is completed, each provider $j$ is rewarded proportionally to the number of times it was selected by the customers:

$$r_j = p \cdot \#\{i\ |\ \tilde{X}_{m_j} \in \mathcal{X}_i\}$$

where $p$ is a constant "price" that scales the reward.

In this experiment we assume the customers **do not know the true likelihoods** $P_{true}(X_m = x | Z = z)$ or the true prior $P(Z | \emptyset)$ (though they start with some initial prior, potentially the true one). Instead, they operate entirely based on their **own likelihood models $L_{i,m,z,x}$** and their evolving beliefs. We also assume that customers do not pool their latent state inferences, i.e. each customer maintains its own independent posterior $P_i(Z | \mathcal{X}_i)$, updated using their $L_{i,m,z,x}$. This reflects a competitive setting where each customer is trying to make the best possible inference about the shared latent variable $Z$ based on their own observations and their own (potentially flawed) model of the world.

#### 1.2.3 Trust Model (Beta Mixture)

To allow for faster adaptation to potential abrupt changes in provider behavior (e.g., a switch to deception), we replace the single Beta distribution with a **two-component Beta mixture model**. Each customer $i$ maintains beliefs about provider $j$ for modality $m$ using:

1.  **Component 1: "Normal Operation" Model:** $P_1(\rho_{jm}) = \text{Beta}(\rho_{jm} | \alpha_{ijm}, \beta_{ijm})$
    *   Represents the accumulated evidence assuming the provider operates consistently (honestly or randomly). Initialized with $\alpha_{ijm}=1, \beta_{ijm}=1$.
2.  **Component 2: "Deceptive Operation" Model:** $P_2(\rho_{jm}) = \text{Beta}(\rho_{jm} | \alpha_D, \beta_D)$
    *   Represents the belief that the provider is actively misleading. This component is *fixed* with parameters strongly favouring low reliability, e.g., $\alpha_D=1, \beta_D=10$.
3.  **Mixture Weight:** $w_{ijm}$
    *   The probability that customer $i$ believes provider $j$ is currently operating according to Component 1 (Normal). $(1-w_{ijm})$ is the belief in Component 2 (Deceptive). Initialized with $w_{ijm} \approx 1$ (e.g., 0.99).

The overall belief distribution is $P_i(\rho_{jm}) = w_{ijm} P_1(\rho_{jm}) + (1-w_{ijm}) P_2(\rho_{jm})$.

**Updating the Model:**

At the end of round $t+1$, after the true state $\hat{z}$ is revealed (for reward/update purposes), each customer $i$ updates their trust model for each provider $j$ they received a measurement $x_{mj}$ from. Crucially, the "correctness" of the observation $x_{mj}$ is evaluated **according to the customer's own likelihood model**, by calculating $L_{i,true} = L_{i, m_j, \hat{z}, x_{mj}}$.

1.  **Update Normal Component Parameters:** These accumulate evidence as before, using the *pooled prior* from the previous round ($\alpha^{(t)}_{\star jm}, \beta^{(t)}_{\star jm}$) and the customer's own assessment of correctness:
    $$ \alpha^{(t+1)}_{ijm} = \alpha^{(t)}_{\star jm} + \sum_{x_{mj} \in \mathcal{X}_{ijm}} L_{i,true} $$
    $$ \beta^{(t+1)}_{ijm} = \beta^{(t)}_{\star jm} + \sum_{x_{mj} \in \mathcal{X}_{ijm}} (1 - L_{i,true}) $$
2.  **Update Mixture Weight:** The weight $w_{ijm}$ is updated using Bayes' theorem based on how well the total evidence from the round ($\Delta\alpha = \sum L_{i,true}$, $\Delta\beta = \sum (1 - L_{i,true})$) fits each component. Let $w^{(t)}_{\star jm}$ be the pooled prior weight.
    *   Calculate the evidence likelihood under each component using the Beta function, $B(x,y)$:
        *   $Ev_1 = \frac{B(\alpha^{(t)}_{\star jm} + \Delta\alpha, \beta^{(t)}_{\star jm} + \Delta\beta)}{B(\alpha^{(t)}_{\star jm}, \beta^{(t)}_{\star jm})}$ (Likelihood under updated Normal model)
        *   $Ev_2 = \frac{B(\alpha_D + \Delta\alpha, \beta_D + \Delta\beta)}{B(\alpha_D, \beta_D)}$ (Likelihood under fixed Deceptive model)
    *   Update the odds: $\frac{w^{(t+1)}_{ijm}}{1-w^{(t+1)}_{ijm}} = \frac{Ev_1}{Ev_2} \cdot \frac{w^{(t)}_{\star jm}}{1-w^{(t)}_{\star jm}}$
    *   Normalize to get the posterior weight $w^{(t+1)}_{ijm}$. (Note: Requires stable computation of log-Beta functions).

**Pooling:**

At the end of the round, parameters and weights are pooled to form the global prior for the next round:

*   $\alpha^{(t+1)}_{\star jm} = \sum_i (\alpha^{(t+1)}_{ijm} - \alpha^{(t)}_{\star jm}) + \alpha^{(t)}_{\star jm}$ (Sum of updates plus prior, effectively summing all evidence)
*   $\beta^{(t+1)}_{\star jm} = \sum_i (\beta^{(t+1)}_{ijm} - \beta^{(t)}_{\star jm}) + \beta^{(t)}_{\star jm}$ (Similarly for beta)
*   $w^{(t+1)}_{\star jm} = \frac{1}{N_{cust}} \sum_i w^{(t+1)}_{ijm}$ (Average the mixture weights)

**Expected Reliability:**

When selecting providers, customers use the expected reliability under the mixture model:

$$ E_i[\rho_{jm}] = w_{ijm} \frac{\alpha_{ijm}}{\alpha_{ijm} + \beta_{ijm}} + (1 - w_{ijm}) \frac{\alpha_D}{\alpha_D + \beta_D} $$

**Rationale and Discussion:**

*   **Handling Sharp Turns:** This mixture model explicitly accounts for the possibility of deceptive behavior. If a provider starts generating observations with very low $L_{i,true}$ (strong evidence against their previously established reliability in Component 1), the evidence likelihood $Ev_1$ will become very small compared to $Ev_2$. This causes the posterior odds to shift rapidly, decreasing $w_{ijm}$ and increasing the weight on the "Deceptive" component. Consequently, the overall expected reliability $E_i[\rho_{jm}]$ drops quickly, allowing the customer to react much faster than the simple Beta model could.
*   **Preserves Accumulated Evidence:** The model retains the history of the provider's "normal" behavior in $\alpha_{ijm}, \beta_{ijm}$. If the provider were to revert to good behavior, $Ev_1$ would start increasing relative to $Ev_2$, and $w_{ijm}$ could potentially recover.
*   **Detecting Deception:** The mixture weight $w_{ijm}$ itself becomes an indicator of suspected deception. Values close to 0 suggest the customer strongly believes the provider is currently deceptive.
*   **Complexity:** This model is more complex computationally (requiring Beta function calculations) and conceptually than the single Beta model but offers significantly better resilience against strategic deception.
*   **Leverages Ground Truth (for state only):** The update relies on the availability of the true latent state $\hat{z}$ for evaluation, but uses the customer's *own* likelihood model $L_{i,m,z,x}$ to interpret the observation relative to that state.

### 1.3 Agent implementations

#### 1.3.1 Provider implementations

We test three provider implementations:

*   **Passive**: A provider that has a true reliability $\hat{\rho}_{jm}$ and always returns measurements sampled according to that reliability. This is a useful baseline that allows us to test whether the consumers can learn the true reliability of the providers.

*   **Active**: A provider that can calibrate its reliability by expending some effort. At each round, the provider can choose to either return a "good" measurement sampled according to its true reliability $\hat{\rho}_{jm}$, paying a cost $c_j$, or to return a random measurement (effectively reliability $1/N_x$) at zero cost. 
    *   We assume the provider knows the reward parameter $p$, the cost $c_j$, and can access the current global expected reliability $\bar{\rho}_{jm} = \alpha_{\star jm} / (\alpha_{\star jm} + \beta_{\star jm})$ based on the pooled posterior from the previous round.
    *   The provider estimates the number of customers expected to select it based on its perceived reliability $\rho$. Let $N_{cust}$ be the total number of customers seeking modality $m$, and $J_m$ be the set of providers for modality $m$. We assume the provider uses the following function $N_{est}(\rho)$, which depends on its own perceived reliability $\rho$ and the current perceived reliabilities $\bar{\rho}_{km}$ of its competitors ($k \in J_m, k \neq j$):
        $$ N_{est}(\rho) = N_{cust} \cdot \frac{\rho}{\sum_{k \in J_m, k \neq j} \bar{\rho}_{km} + \rho} $$
    *   **Decision Rule:** The provider compares the expected net benefit of providing a Good Measurement vs. a Random Measurement. It chooses to provide a **Good Measurement** (paying cost $c_j$) if the expected *increase* in reward due to maintaining its current reputation $\bar{\rho}_{jm}$ (compared to the baseline reputation $1/N_x$ it would eventually acquire by always providing random measurements) exceeds the cost:
        $$ p \cdot [N_{est}(\bar{\rho}_{jm}) - N_{est}(1/N_x)] > c_j $$
        Otherwise, it provides a **Random Measurement** (cost 0). This reflects the incentive to maintain a good reputation only if the resulting increase in expected selections justifies the cost.
    *   **Expected Dynamics and Sorting:** This decision rule is expected to create a sorting equilibrium based on a provider's intrinsic quality ($\hat{\rho}_{jm}$) and cost ($c_j$), mediated by the market parameters ($p, N_{cust}$) and competition ($\sum \bar{\rho}_{km}$). 
        *   **High Quality/Low Cost Providers:** Providers with a sufficiently high true reliability $\hat{\rho}_{jm}$ and/or a sufficiently low cost $c_j$ will likely find that the expected reputational benefit $p \cdot [N_{est}(\bar{\rho}_{jm}) - N_{est}(1/N_x)]$ exceeds $c_j$, incentivizing them to provide **Good Measurements**. When they do so, customer updates will tend to push their perceived reliability $\bar{\rho}_{jm}$ towards their true reliability $\hat{\rho}_{jm}$. As $\bar{\rho}_{jm}$ increases, $N_{est}(\bar{\rho}_{jm})$ generally increases, reinforcing the incentive to provide Good Measurements. This creates a positive feedback loop, leading these providers to consistently provide high-quality information and their reputation $\bar{\rho}_{jm}$ to converge towards their true $\hat{\rho}_{jm}$.
        *   **Low Quality/High Cost Providers:** Conversely, providers with low $\hat{\rho}_{jm}$ or high $c_j$ may find the inequality does not hold, leading them to provide **Random Measurements**. Customer updates will then push their perceived reliability $\bar{\rho}_{jm}$ towards the baseline $1/N_x$. As $\bar{\rho}_{jm}$ approaches $1/N_x$, the term $[N_{est}(\bar{\rho}_{jm}) - N_{est}(1/N_x)]$ approaches zero, further reducing the incentive to pay the cost $c_j$. This feedback loop leads these providers to consistently provide uninformative measurements, and their reputation $\bar{\rho}_{jm}$ converges towards $1/N_x$.
        *   **Competition:** The presence of strong competitors (high $\sum_{k \neq j} \bar{\rho}_{km}$) reduces $N_{est}(\rho)$ for any given $\rho$, making it harder for any provider to justify the cost $c_j$. Therefore, the threshold for switching between 'Good' and 'Random' behavior depends on the competitive landscape.
        *   Overall, the system should sort providers based on their cost-effectiveness, with customers learning to trust and utilize providers whose true quality justifies their operational cost within the given market structure.

*   **Strategic (Threshold Switcher)**: A provider acting to maximize long-term discounted rewards, facing a one-time, irreversible decision to switch from costly honesty to potentially rewarding deception. This sharply tests incentive alignment.
    *   **Capabilities & Parameters:**
        *   True reliability when Honest: $\hat{\rho}_{jm}$.
        *   Cost per round when Honest: $c_j$.
        *   Reward factor when Honest: $p$.
        *   Reward factor when Deceptive: $p_{lie}$.
        *   Discount factor: $\delta \in (0, 1]$.
        *   Anticipated reputation decay period: $\tau_{decay}$.
    *   **State & Actions:** Starts 'Honest'. Each round $t$ while 'Honest', chooses:
        1.  **Stay Honest:** Pay $c_j$, provide Good measurement (using $\hat{\rho}_{jm}$), remain 'Honest'.
        2.  **Switch to Deceptive:** Pay 0, provide Misleading measurement ($L_{i,true} \approx 0$), permanently become 'Deceptive'.
        If 'Deceptive', always provides Misleading measurement at cost 0.
    *   **NPV-Based Decision Rule (with Anticipated Decay):** The provider performs a forward-looking calculation at the start of round $t$, comparing the expected NPV of staying honest vs. switching now. Crucially, it anticipates that if it switches, its reputation (and thus selection rate $N_{est}$) will not instantly collapse but will decay over an estimated period of $\tau_{decay}$ rounds.
        1.  **NPV(Stay Honest):** Calculated as the immediate net reward $[p \cdot N_{est}(\bar{\rho}_{jm, t}, t) - c_j]$ plus the discounted expected future value assuming it continues to act honestly (approximated by the steady-state value of perpetual honesty, $V_{Honest}^{steady} = \frac{p \cdot N_{Honest} - c_j}{1-\delta}$).
        2.  **NPV(Switch Now):** Calculated by summing discounted expected rewards:
            *   *During Decay (rounds $t$ to $t+\tau_{decay}-1$):* The provider receives reward $p_{lie}$ per selection and assumes its selection rate remains constant at the current level, $N_{est}(\bar{\rho}_{jm, t}, t)$. The discounted sum of these rewards is $p_{lie} \cdot N_{est}(\bar{\rho}_{jm, t}, t) \cdot \frac{1 - \delta^{\tau_{decay}}}{1-\delta}$.
            *   *Post Decay (rounds $t+\tau_{decay}$ onwards):* After the decay period, the provider receives the discounted steady-state value of perpetual deception. This value, discounted back to round $t$, is $\delta^{\tau_{decay}} \cdot V_{Deceptive}^{steady} = \delta^{\tau_{decay}} \cdot \frac{p_{lie} \cdot N_{Deceptive}}{1-\delta}$.
            *   The total $NPV(Switch Now)$ is the sum of the *During Decay* and *Post Decay* discounted values.
        *   **Switch Condition:** The provider calculates the expected NPV for both paths and chooses to **Switch to Deceptive** if $NPV(Switch Now) > NPV(Stay Honest)$.
    *   **Interpretation & Test Focus:**
        *   **Sophistication:** This rule models a more sophisticated adversary who anticipates market reaction delays ($\tau_{decay}$). A longer anticipated decay period makes switching more attractive, as the provider expects to profit from deception for longer before facing consequences.
        *   **Cooption vs. Defection:** The rule balances immediate temptation against the long-term value of reputation. The provider is coopted into perpetual honesty if the opportunity cost (right side) always exceeds the potential immediate gain (left side). Defection (switching) occurs when the immediate gain becomes sufficiently attractive relative to the future value lost.
        *   **Parameter Sensitivity:** Tests how $\tau_{decay}$ interacts with $\hat{\rho}_{jm}, c_j, p, p_{lie}, \delta$ to influence the decision. Does a faster expected decay (smaller $\tau_{decay}$) promote honesty?
        *   **Clean Test:** Provides a framework for analyzing incentive alignment against strategic agents with expectations about market response times.

#### 1.3.2 Customer implementations (Active Learning)

Customers implement an active learning strategy to iteratively select providers within each round, aiming to maximize the expected information gain about the latent variable $Z$. The core mechanism follows the description in Section 1.2.2:

1.  **Maintain Beliefs:** Each customer $i$ maintains a posterior $P_i(Z | \mathcal{X}_i)$ over the latent variable $Z$, initialized with a prior $P(Z | \emptyset)$ and updated using their **own likelihood model $L_{i,m,z,x}$**. They also maintain reliability beliefs $P_i(\rho_{jm})$ for each provider $j$ and modality $m$ using the **Trust Model** defined in Section 1.2.3.
2.  **Calculate Expected Information Gain:** For each provider $j$ whose modality $m_j$ observation has not yet been added to the current set $\mathcal{X}_i$, the customer calculates the expected increase in reward (expected KL divergence) $EIG_j$ from adding that provider's *soft observation* $\tilde{X}_{m_j}$. 
    *   The reliability estimate $\rho_{ij}$ used to construct the soft likelihood $\tilde{L}_{ij}$ (using the customer's $L_{i,m_j}$) is derived from the customer's current trust model $P_i(\rho_{jm})$.
    *   The formula for $EIG_j$ is:
        $$ EIG_j = H(P_i) - (p_{ij}^x)^T H_{i,Z|X,j} $$
3.  **Iterative Selection:** The customer selects the provider $j^*$ with the highest positive expected gain $EIG_{j^*} > 0$ and adds their (conceptual) soft observation $\tilde{X}_{m_{j^*}}$ to $\mathcal{X}_i$, updating the posterior $P_i(Z | \mathcal{X}_i)$.
4.  **Stopping Criteria:** This selection process repeats until one of the stopping conditions is met (e.g., information budget depleted, max $EIG_j \le 0$, posterior entropy threshold reached).

**Variations in Customer Strategy:**

While the core framework is active learning, customer implementations can differ primarily in how they **derive the reliability estimate $\rho_{jmi}$** used in the $EIG_j$ calculation from their trust model $P_i(\rho_{jm})$, or in their **stopping criteria**:

*   **Standard Active Learner (Mean Reliability):**
    *   **Reliability Use:** Uses the *mean* expected reliability $\rho_{jmi} = E_i[\rho_{jm}]$ derived from the Beta mixture model when calculating $\tilde{L}$ and $EIG_j$.
    *   **Purpose:** Represents the standard Bayesian active learning approach, selecting based purely on expected value according to the current mean belief.
*   **Thompson Sampling Active Learner:**
    *   **Reliability Use:** When evaluating the potential gain $EIG_j$ for each candidate provider $j$ in a selection step, draws a *sample* $\tilde{\rho}_{jm}$ from the current posterior $P_i(\rho_{jm})$. Uses this *sampled* reliability $\tilde{\rho}_{jm}$ to calculate $\tilde{L}$ and the resulting $EIG_j$. Selects the provider maximizing $EIG_j$ based on these sampled values.
    *   **Purpose:** Integrates exploration directly into the information value assessment. Providers with uncertain but potentially high reliability might be chosen over providers with known mediocre reliability, promoting faster learning.
*   **UCB Active Learner:**
    *   **Reliability Use:** Uses an optimistic reliability estimate $\rho_{jmi} = E_i[\rho_{jm}] + \beta$ (where the bonus $\beta$ depends on uncertainty, e.g., $c \cdot \sqrt{\frac{\log t}{N_{ijm}}}$) when calculating $\tilde{L}$ and $EIG_j$.
    *   **Purpose:** Explicitly inflates the perceived value of less-certain providers to encourage exploration.
*   **Variations in Stopping Criteria:** Customers could also differ in their budget size, their information gain threshold ($EIG_j > \epsilon$ instead of $EIG_j > 0$), or their target posterior entropy level.

### 1.4 Simulation Parameters

*   **General:**
    *   Number of Rounds: $T$ (e.g., 500)
    *   Number of Customers: $N_C$ (e.g., 10)
    *   Number of Providers: $N_P$ (e.g., 20)
    *   Number of Modalities: $N_M$ (e.g., 5)
    *   Provider-Modality Mapping: Specify which provider(s) offer which modality (e.g., random assignment, specific overlaps).
*   **World Model:**
    *   Latent Variable $Z$: Number of states, true prior distribution $P(Z=z)$.
    *   True Likelihood $P_{true}(X_m=x | Z=z)$: The actual data generating process.
    *   Customer Likelihood Models $L_{i,m,z,x}$: Distribution of potentially incorrect likelihoods used by customers. Number of observable states per modality $\#\{x\}$.
*   **Provider Parameters:**
    *   Simple Providers: True reliability when Honest: $\hat{\rho}_{jm}$.
    *   Strategic Providers: Base reliability $\hat{\rho}_{jm}$, cost $c_j$, lie probability $p_{lie}$, benefit $\delta$, decay estimate $\tau_{decay}$. Distribution of provider types.
*   **Trust Model Parameters:**
    *   Initial Beta Parameters: $\alpha_0, \beta_0$ for the normal component prior.
    *   Deceptive Component: Fixed parameters $\alpha_D, \beta_D$ (e.g., Beta(1, 9)).
    *   Initial Mixture Weight: $w_{ijm}(0)$. 
    *   Pooling mechanism (if applicable, e.g., averaging weights/parameters across customers).
*   **Customer Parameters:**
    *   Active Learning Stopping Criteria: Information budget (max selections), $EIG_j$ threshold $\epsilon$, posterior entropy threshold $H_{min}$.
    *   UCB Constant: $c$ (if using UCB Active Learner).
*   **Reward Parameters:**
    *   Provider Reward Factor: $p$ (scaling factor for reward per selection).

### 1.5 Metrics and Evaluation Criteria

*   **Customer Performance:**
    *   Average Customer Reward: Mean $r_i = \log P_i(Z = \hat{z} | \mathcal{X}_i)$ across customers over time (Note: reward is based on customer's own posterior evaluated at the true state).
    *   Posterior Convergence: Rate at which $P_i(Z | \mathcal{X}_i)$ converges towards a distribution concentrated on the true latent state $\hat{z}$ (convergence might be imperfect due to incorrect $L_i$).
    *   Final Posterior Entropy: Mean $H(P_i(Z | \mathcal{X}_i))$ at the end of the simulation.
    *   Information Budget Used: Average number of providers selected per customer per round.
*   **Trust Model Accuracy:**
    *   Reliability Estimation Error: Mean Squared Error (MSE) between $E_i[\rho_{jm}]$ and true $\rho_{jm}$.
    *   Deception Detection Speed: Number of rounds until the mixture weight $w_{ijm}$ for a deceptive provider drops below a threshold (e.g., 0.5).
*   **Provider Sorting & Market Dynamics:**
    *   Reward Correlation: Spearman correlation between provider true utility/reliability and total earned rewards $r_j$.
    *   Market Share: Distribution of selections across providers over time. Does selection frequency correlate with true quality?
*   **System Efficiency:**
    *   Aggregate Reward: Sum of rewards for all agents (customers and providers).
    *   Information Quality: Ratio of useful information acquired (contributing to correct inference) vs. total information paid for.

### 1.6 Detailed Hypotheses and Experimental Questions

This section breaks down the high-level objectives from Section 1.1 into specific, testable questions.

**H1: Trust modeling can help nodes make better decisions by grounding their inferences in reliable data.**

*   **H1.1:** Customers using active learning based on reliability estimates (Standard, Thompson, UCB) will achieve significantly higher average rewards ($r_i$) than baseline Uniform customers who ignore reliability.
*   **H1.2:** Thompson Sampling Active Learners will achieve higher average rewards and/or faster convergence than Standard (Mean Reliability) Active Learners, especially when provider reliabilities vary or change over time.
*   **Q1.1:** How does the performance of UCB Active Learners compare to Thompson Sampling and Standard learners? How sensitive is UCB to the choice of exploration constant $c$?
*   **Q1.2:** How does the information budget size or other stopping criteria affect the final inference quality and cost for different customer types?

**H2: Incentives can further improve the quality of data provided by nodes.**

*   **H2.1:** The presence of rewards ($p > 0$) linked to selection incentivizes Simple Providers with higher true reliability $\rho_{jm}$ to participate and earn more.
*   **H2.2:** Strategic Providers will choose to remain honest ($\text{provide } \hat{\rho}_{jm}$ at cost $c_j$) only when the expected long-term reward from maintaining a good reputation outweighs the short-term gain from deception, considering the reward factor $p$ and the estimated decay $\tau_{decay}$.
*   **Q2.1:** What is the parameter space (cost $c_j$, benefit $\delta$, reward $p$, decay $\tau_{decay}$, true reliability $\hat{\rho}_{jm}$) where Strategic Providers are incentivized to be honest?
*   **Q2.2:** How does competition (number of providers per modality) affect the reward dynamics and provider behavior?

**H3: The basic setup is resilient against spam and malicious nodes.**

*   **H3.1:** The Beta Mixture trust model will allow customers (Standard, Thompson, UCB) to rapidly downgrade their reliability estimates ($E_i[\rho_{jm}]$ and mixture weight $w_{ijm}$) for a Strategic Provider after it switches to deception.
*   **H3.2:** Customers using active learning will quickly reduce their selection frequency of a provider identified as deceptive, minimizing the negative impact on their inference compared to a Uniform customer.
*   **Q3.1:** How quickly does the system (reliability estimates and provider selections) react to a Strategic Provider switching compared to the provider's expectation ($\tau_{decay}$)?
*   **Q3.2:** Can a Strategic Provider successfully perform a "bait-and-switch" (build reputation, then deceive) against different customer types? How effective is it?
*   **Q3.3:** How does the performance degrade with an increasing proportion of low-reliability or Strategic (deceptive) providers?

## Experiment 2: Additional mechanisms (LATER)

In this experiment, we will explore whether a combination of mechanisms such as escrows, endorsements and trust staking can further improve the resilience and overall quality of the network.
