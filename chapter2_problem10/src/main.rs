extern crate rand;
extern crate regex;

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use rand::Rng;
use regex::Regex;

#[derive(Debug)]
pub struct HmmModel {
    state_transition_matrix: Vec<Vec<f64>>,
    observation_probability_matrix: Vec<Vec<f64>>,
    initial_state_distribution_vector: Vec<f64>,
}

impl HmmModel {
    pub fn train_model(number_of_hidden_state_symbols: usize, number_of_observation_symbols: usize, observations: &Vec<usize>) -> HmmModel {
        let min_initial_value = 30.0;
        let max_initial_value = 70.0;
        let min_iterations = 100;
        let max_iterations = 250;
        let improvement_threshold = 0.01;

        let mut model = HmmModel {
            state_transition_matrix: vec![Vec::with_capacity(number_of_hidden_state_symbols); number_of_hidden_state_symbols],
            observation_probability_matrix: vec![Vec::with_capacity(number_of_observation_symbols); number_of_hidden_state_symbols],
            initial_state_distribution_vector: Vec::with_capacity(number_of_hidden_state_symbols),
        };

        // Generate guesses
        let mut rng = rand::thread_rng();
        for i in 0..number_of_hidden_state_symbols {
            for j in 0..number_of_hidden_state_symbols {
                model.state_transition_matrix[i].insert(j, rng.gen_range(min_initial_value, max_initial_value));
            }

            for j in 0..number_of_observation_symbols {
                model.observation_probability_matrix[i].insert(j, rng.gen_range(min_initial_value, max_initial_value));
            }

            model.initial_state_distribution_vector.insert(i, rng.gen_range(min_initial_value, max_initial_value));
        }

        // Normalize
        for i in 0..number_of_hidden_state_symbols {
            let state_transition_row_sum = model.state_transition_matrix[i].iter().sum::<f64>();
            model.state_transition_matrix[i] = model.state_transition_matrix[i].iter().map(|&probability| probability / state_transition_row_sum).collect();

            let observation_probability_row_sum = model.observation_probability_matrix[i].iter().sum::<f64>();
            model.observation_probability_matrix[i] = model.observation_probability_matrix[i].iter().map(|&probability| probability / observation_probability_row_sum).collect();
        }
        let initial_state_distribution_vector_sum = model.initial_state_distribution_vector.iter().sum::<f64>();
        model.initial_state_distribution_vector = model.initial_state_distribution_vector.iter().map(|&probability| probability / initial_state_distribution_vector_sum).collect();

        println!("Initial model: {:?}", &model);

        let mut iterations = 0;
        let mut log_probability = std::f64::NEG_INFINITY;
        let mut old_log_probability = std::f64::NEG_INFINITY;

        while iterations < min_iterations || (log_probability - old_log_probability).abs() > improvement_threshold && iterations < max_iterations {
            old_log_probability = log_probability;

            let (alpha_matrix, scale_factors) = model.compute_alpha_matrix_and_scale_factors(&observations);
            let beta_matrix = model.compute_beta_matrix(&observations, &scale_factors);
            let (gamma_matrix, di_gamma_tensor) = model.compute_gamma_matrix_and_di_gamma_tensor(&observations, &alpha_matrix, &beta_matrix);

            model.initial_state_distribution_vector = gamma_matrix[0].to_vec();
            for i in 0..number_of_hidden_state_symbols {
                for j in 0..number_of_hidden_state_symbols {
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;
                    for observation_index in 0..(observations.len() - 1) {
                        numerator += di_gamma_tensor[observation_index][i][j];
                        denominator += gamma_matrix[observation_index][i];
                    }

                    model.state_transition_matrix[i][j] = numerator / denominator;
                }

                for j in 0..number_of_observation_symbols {
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;
                    for observation_index in 0..(observations.len() - 1) {
                        if observations[observation_index] == j {
                            numerator += gamma_matrix[observation_index][i];
                        }
                        denominator += gamma_matrix[observation_index][i];
                    }

                    model.observation_probability_matrix[i][j] = numerator / denominator;
                }
            }

            log_probability = -(scale_factors.iter().map(|&scalar| scalar.log2())).sum::<f64>();
            println!("Finished iteration {:?}", iterations);
            iterations += 1;
        }

        println!("Done training.");
        println!("Final model: {:?}", &model);

        return model;
    }

    pub fn get_number_of_hidden_state_symbols(&self) -> usize {
        return self.initial_state_distribution_vector.len();
    }

    pub fn get_number_of_observation_symbols(&self) -> usize {
        return match self.observation_probability_matrix.len() {
            0 => 0,
            _ => self.observation_probability_matrix[0].len(),
        };
    }

    fn compute_alpha_matrix_and_scale_factors(&self, observations: &Vec<usize>) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut alpha_matrix = vec![
            vec![0.0; self.get_number_of_hidden_state_symbols()]; observations.len()
        ];
        let mut scale_factors = vec![0.0; observations.len()];

        for i in 0..self.get_number_of_hidden_state_symbols() {
            // alpha_0(i) = pi_i * b_i(O_0)
            alpha_matrix[0][i] = self.initial_state_distribution_vector[i] * self.observation_probability_matrix[i][observations[0]];
            scale_factors[0] += alpha_matrix[0][i]
        }

        scale_factors[0] = 1.0 / scale_factors[0];
        for i in 0..self.get_number_of_hidden_state_symbols() {
            alpha_matrix[0][i] *= scale_factors[0];
        }

        for observation_index in 1..observations.len() {
            for i in 0..self.get_number_of_hidden_state_symbols() {
                for j in 0..self.get_number_of_hidden_state_symbols() {
                    // += alpha_t-1(j) * a_ji
                    alpha_matrix[observation_index][i] += alpha_matrix[observation_index - 1][j] * self.state_transition_matrix[j][i];
                }

                // = sum(alpha_t-1(j) * a_ji) * b_i(O_t)
                alpha_matrix[observation_index][i] *= self.observation_probability_matrix[i][observations[observation_index]];
                scale_factors[observation_index] += alpha_matrix[observation_index][i];
            }


            scale_factors[observation_index] = 1.0 / scale_factors[observation_index];
            for i in 0..self.get_number_of_hidden_state_symbols() {
                alpha_matrix[observation_index][i] *= scale_factors[observation_index];
            }
        }

        return (alpha_matrix, scale_factors);
    }

    fn compute_beta_matrix(&self, observations: &Vec<usize>, scale_factors: &Vec<f64>) -> Vec<Vec<f64>> {
        let mut beta_matrix = vec![
            vec![0.0; self.get_number_of_hidden_state_symbols()]; observations.len()
        ];

        // beta_T-1(i) = c_T-1
        beta_matrix[observations.len() - 1] = vec![scale_factors[observations.len() - 1]; self.get_number_of_hidden_state_symbols()];

        // From T-2 to 0
        for observation_index in (0..(observations.len() - 1)).rev() {
            for i in 0..self.get_number_of_hidden_state_symbols() {
                for j in 0..self.get_number_of_hidden_state_symbols() {
                    // += a_ij * b_j(O_t+1) * beta_t+1(j)
                    beta_matrix[observation_index][i] += self.state_transition_matrix[i][j] * self.observation_probability_matrix[j][observations[observation_index + 1]] * beta_matrix[observation_index + 1][j];
                }

                beta_matrix[observation_index][i] *= scale_factors[observation_index];
            }
        }

        return beta_matrix;
    }

    fn compute_gamma_matrix_and_di_gamma_tensor(&self, observations: &Vec<usize>, alpha_matrix: &Vec<Vec<f64>>, beta_matrix: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let mut gamma_matrix = vec![
            vec![0.0; self.get_number_of_hidden_state_symbols()]; observations.len()
        ];
        let mut di_gamma_tensor = vec![
            vec![Vec::with_capacity(self.get_number_of_hidden_state_symbols()); self.get_number_of_hidden_state_symbols()]; observations.len()
        ];

        // From 0 to T-2
        for observation_index in 0..(observations.len() - 1) {
            let mut denominator = 0.0;
            for i in 0..self.get_number_of_hidden_state_symbols() {
                for j in 0..self.get_number_of_hidden_state_symbols() {
                    // += alpha_t(t) * a_ij * b_j(O_t+1) * beta_t+1(j)
                    denominator += alpha_matrix[observation_index][i] * self.state_transition_matrix[i][j] * self.observation_probability_matrix[j][observations[observation_index + 1]] * beta_matrix[observation_index + 1][j];
                }
            }

            for i in 0..self.get_number_of_hidden_state_symbols() {
                for j in 0..self.get_number_of_hidden_state_symbols() {
                    // += (alpha_t(i) * a_ij * b_j(O_t+1) * beta_t+1(j)) / denom
                    di_gamma_tensor[observation_index][i].insert(j, (alpha_matrix[observation_index][i] * self.state_transition_matrix[i][j] * self.observation_probability_matrix[j][observations[observation_index + 1]] * beta_matrix[observation_index + 1][j]) / denominator);
                    // += di-gamma_t(i, j)
                    gamma_matrix[observation_index][i] += di_gamma_tensor[observation_index][i][j];
                }
            }
        }

        let denominator = alpha_matrix[observations.len() - 1].iter().sum::<f64>();
        gamma_matrix[observations.len() - 1] = alpha_matrix[observations.len() - 1].iter().map(|&alpha_value| alpha_value / denominator).collect();

        return (gamma_matrix, di_gamma_tensor);
    }
}

fn main() {
    let number_of_observation_symbols = 27;

    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        print_usage_and_panic();
    }
    let number_of_hidden_state_symbols = args[1].parse::<usize>().unwrap();
    let filename = &args[2];

    let path = Path::new(filename);

    let mut file = match File::open(&path) {
        Err(reason) => panic!("Couldn't open file: {:?}", reason),
        Ok(file) => file,
    };

    let mut raw_input = String::new();
    file.read_to_string(&mut raw_input).unwrap();

    let lowercase_input = raw_input.to_lowercase();
    let no_extra_spaces_no_new_lines = Regex::new("(\n\\s*)|(\\s+\\s+)").unwrap().replace_all(lowercase_input.as_str(), " ").to_string();
    let sanitized_input = Regex::new("[^a-z ]").unwrap().replace_all(no_extra_spaces_no_new_lines.as_str(), "");

    // z, y, x, ... a, SPACE => 0, 1, 2, ..., 25, 26
    let observations = sanitized_input.chars().map(|ch| match ch {
        ' ' => 26 as usize,
        _ => ch as usize - 'a' as usize,
    }).collect();

    println!("raw_input: {:?}", raw_input);
    println!("sanitized_input: {:?}", sanitized_input);
    println!("observations: {:?}", observations);

    let model = HmmModel::train_model(number_of_hidden_state_symbols, number_of_observation_symbols, &observations);

    for j in 0..number_of_observation_symbols {
        match j {
            26 => print!("SPACE  "),
            _ => print!("{:?}      ", (j as u8 + 'a' as u8) as char),
        }

        let mut state_max_probability = (0, 0.0);
        for i in 0..number_of_hidden_state_symbols {
            print!("{:.*}   ", 5, model.observation_probability_matrix[i][j]);

            if model.observation_probability_matrix[i][j] > state_max_probability.1 {
                state_max_probability = (i, model.observation_probability_matrix[i][j]);
            }
        }
        println!("{:?}", state_max_probability.0);
    }
}

fn print_usage_and_panic() {
    println!("Usage: cargo run <number_of_hidden_state_symbols> <input_file>");
    panic!("Incorrect command arguments");
}
