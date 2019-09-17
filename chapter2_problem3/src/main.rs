use std::time::SystemTime;

fn main() {
    let mut test = vec![0.0; 1234567890];
    println!("created");
    let time_before = SystemTime::now();
    let slice = &test[2..123456789].iter_mut().for_each(|x: &mut f64| x = 1.0);
    for i in 2..123456789 {
        test[i] = 1.0;
    }
    println!("mutated after: {:.*}s", 3, time_before.elapsed().unwrap().as_millis() as f64 / 1000.0);
    println!("{:?}", test[2]);
    println!("length: {:?}", test.len());
//
//    let state_transition_matrix = [[0.7, 0.3], [0.4, 0.6]];
//    let observation_probability_matrix = [[0.1, 0.4, 0.5], [0.7, 0.2, 0.1]];
//    let initial_state_distribution = [0.6, 0.4];
//
//    let number_of_hidden_state_symbols = initial_state_distribution.len();
//    let number_of_observation_symbols = observation_probability_matrix[0].len();
//    let number_of_observations = 4 as usize;
//
//    let total_number_of_hidden_states = (number_of_hidden_state_symbols as f64).powf(number_of_observations as f64);
//    let total_number_of_observation_states = (number_of_observation_symbols as f64).powf(number_of_observations as f64);
//
//    let mut sum_of_all_probabilities_direct_calculation = 0 as f64;
//    let mut sum_of_all_probabilities_forward_algorithm = 0 as f64;
//    for current_observation_state_number in 0..(total_number_of_observation_states as i64) {
//        // Make observation ex. 75 = 2*3^3 + 2*3^2 + 2*3^1 + 0*3^0 = [2, 2, 1, 0]
//        let mut observations = Vec::new() as Vec<usize>;
//        let mut n = current_observation_state_number as usize;
//        for _ in 0..number_of_observations {
//            observations.insert(0, n % number_of_observation_symbols);
//            n /= number_of_observation_symbols;
//        }
//
//        // Direct calculation
//        for current_hidden_state_number in 0..(total_number_of_hidden_states as i64) {
//            let mut hidden_states = Vec::new() as Vec<usize>;
//            let mut n = current_hidden_state_number as usize;
//            for _ in 0..number_of_observations {
//                hidden_states.insert(0, n % number_of_hidden_state_symbols);
//                n /= number_of_hidden_state_symbols;
//            }
//
//            // = pi_x_0
//            let mut hidden_states_probability = initial_state_distribution[hidden_states[0]];
//            for hidden_state_pair in (&hidden_states[1..]).iter().enumerate() {
//                let prev_hidden_state = hidden_states[hidden_state_pair.0];
//                let observation = observations[hidden_state_pair.0];
//
//                // *= b_x_i(O_0)
//                hidden_states_probability *= observation_probability_matrix[prev_hidden_state][observation];
//                // *= a_x_i, a_x_i+1
//                hidden_states_probability *= state_transition_matrix[prev_hidden_state][*hidden_state_pair.1];
//            }
//
//            hidden_states_probability *= observation_probability_matrix[hidden_states[hidden_states.len() - 1]][observations[observations.len() - 1]];
//
//            sum_of_all_probabilities_direct_calculation += hidden_states_probability;
//            println!("Hidden_states: {:?}", hidden_states);
//            println!("Hidden_states_probability: {:?}", hidden_states_probability);
//        }
//
//        // Forward algorithm
//        let mut alpha_matrix = Vec::new();
//        for i in 0..number_of_hidden_state_symbols {
//            // alpha_0 = pi_i * b_i(O_0)
//            alpha_matrix.push(initial_state_distribution[i] * observation_probability_matrix[i][observations[0]]);
//        }
//        for observation_index in 1..number_of_observations {
//            let mut new_alpha_matrix = alpha_matrix.to_vec();
//            for i in 0..number_of_hidden_state_symbols {
//                let mut alpha_observation_probability = 0 as f64;
//                for j in 0..number_of_hidden_state_symbols {
//                    // += alpha_t-1(j) * a_ji
//                    alpha_observation_probability += alpha_matrix[j] * state_transition_matrix[j][i];
//                }
//                // = sum(alpha_t-1(j) * a_ji) * b_i(O_t)
//                new_alpha_matrix[i] = alpha_observation_probability * observation_probability_matrix[i][observations[observation_index]];
//            }
//            alpha_matrix = new_alpha_matrix;
//        }
//        for probability in alpha_matrix.iter() {
//            sum_of_all_probabilities_forward_algorithm += *probability;
//        }
//
//        println!("Observations: {:?}", observations);
//        println!("Current probability sum (direct calculation): {:?}", sum_of_all_probabilities_direct_calculation);
//        println!("Current probability sum (forward algorithm): {:?}", sum_of_all_probabilities_forward_algorithm);
//        println!();
//
//    }
//    println!("The sum of all of the probabilities of possible observations given the model is: {:?}", sum_of_all_probabilities_direct_calculation);

}
