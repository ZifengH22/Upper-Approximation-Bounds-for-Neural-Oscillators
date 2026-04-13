1.Run Paper1_earthquake_acc_sample_simulation.m to simulate sample of earthquake excitations
2.Run Paper2_structural_response_sample_simulation.m to numerically solve the ODE to obtain response X_200
3.RUn Paper3_preparing_data_for_training.m to prepare the data for training the neural oscillator
4.Run Paper4_checking_data_for_training.m to the check the prepared data 
5.Run Main_program_RK2NN with self.layer_width = 2, 5, 10, 20, 30, 40, 60 to train the neural oscillator
6.Run Result_display_only_response with self.layer_width = 2, 5, 10, 20, 30, 40, 60 to calculate the responses predicted by the trained models