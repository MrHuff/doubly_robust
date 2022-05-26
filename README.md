# Doubly Robust Counterfactual Mean Embeddings and Testing for Distributional Causal Effects

Accompanying code for the paper Doubly Robust Counterfactual Mean Embeddings and Testing for Distributional Causal Effects

There are many experiments in this repo.

1. Expectation experiments recreate Figure 3 in the paper
2. One can generate all datasets by running data_generator.py or data_generator_strong_confounding.py for the datasets with strong confounding
3. One can use generate_job_parameters to produce the jobs needed to recreate Figure 4 and 5. 
4. The bandit experiments are run by:
   1. domain_shift_simulation.py
   2. context_dim_experiment.py
   3. item_size_experiment.py
   4. recommendation_size_experiment.py
   5. sample_size_experiment.py
   6. user_size_experiment.py
   7. Figure 7 is recreated by plot_bandit_experiment.py

