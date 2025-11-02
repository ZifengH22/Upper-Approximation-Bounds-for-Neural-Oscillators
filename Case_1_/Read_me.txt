1. Run "Paper1_force_correlation_sample_simulation.m" to generate random excitaion samples
2. Run "Paper2_structural_response_sample_simulation" to calculate response samples
3. Run "Paper3_preparing_data_for_training.m" to prepare data for training neural oscillator
4. Run "Main_program_RK2NN_L8" to train neural oscillator. In this step, set self.hidden_number in Args to 1, 2, 3, 4, and 5, respectively. For each setting, run “Main_program_RK2NN_L8” to obtain a neural oscillator with an MLP Pi configured with the corresponding number of hidden layers.
5. Run "Result_display_only_response_L8" to calculate the response samples predicted by the trained neural oscillator. In this step, set self.hidden_number in Args to 1, 2, 3, 4, and 5, respectively. For each setting, run "Result_display_only_response_L8" to simulate response samples using the corresponding trained neural oscillator.
