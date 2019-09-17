extern crate rand;
extern crate regex;

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::SystemTime;

use rand::Rng;
use regex::Regex;

#[derive(Debug)]
pub struct HmmModel {
    pub state_transition_matrix: Box<[Box<[f64]>]>,
    pub observation_probability_matrix: Box<[Box<[f64]>]>,
    pub initial_state_distribution_vector: Box<[f64]>,
    alpha_matrix: Box<[Box<[f64]>]>,
    beta_matrix: Box<[Box<[f64]>]>,
    gamma_matrix: Box<[Box<[f64]>]>,
    di_gamma_tensor: Box<[Box<[Box<[f64]>]>]>,
    scale_factors: Box<[f64]>,
}

impl HmmModel {
    pub fn train_model(number_of_hidden_state_symbols: usize, number_of_observation_symbols: usize, observations: &Box<[usize]>) -> HmmModel {
        let min_initial_value = 30.0;
        let max_initial_value = 70.0;
        let min_iterations = 100;
        let max_iterations = 100;
        let improvement_threshold = 0.01;

        let mut model = HmmModel {
            state_transition_matrix: vec![
                vec![0.0;
                     number_of_hidden_state_symbols
                ].into_boxed_slice();
                number_of_hidden_state_symbols
            ].into_boxed_slice(),
            observation_probability_matrix: vec![
                vec![0.0;
                     number_of_observation_symbols
                ].into_boxed_slice();
                number_of_hidden_state_symbols
            ].into_boxed_slice(),
            initial_state_distribution_vector: vec![0.0; number_of_hidden_state_symbols].into_boxed_slice(),
            alpha_matrix: vec![
                vec![0.0;
                     number_of_hidden_state_symbols
                ].into_boxed_slice();
                observations.len()
            ].into_boxed_slice(),
            beta_matrix: vec![
                vec![0.0;
                     number_of_hidden_state_symbols
                ].into_boxed_slice();
                observations.len()
            ].into_boxed_slice(),
            gamma_matrix: vec![
                vec![0.0;
                     number_of_hidden_state_symbols
                ].into_boxed_slice();
                observations.len()
            ].into_boxed_slice(),
            di_gamma_tensor: vec![
                vec![
                    vec![0.0;
                         number_of_hidden_state_symbols
                    ].into_boxed_slice();
                    number_of_hidden_state_symbols
                ].into_boxed_slice();
                observations.len()
            ].into_boxed_slice(),
            scale_factors: vec![0.0; observations.len()].into_boxed_slice(),
        };

        // Generate guesses
        let mut rng = rand::thread_rng();
        for i in 0..number_of_hidden_state_symbols {
            for j in 0..number_of_hidden_state_symbols {
                model.state_transition_matrix[i][j] = rng.gen_range(min_initial_value, max_initial_value);
            }

            for j in 0..number_of_observation_symbols {
                model.observation_probability_matrix[i][j] = rng.gen_range(min_initial_value, max_initial_value);
            }

            model.initial_state_distribution_vector[i] = rng.gen_range(min_initial_value, max_initial_value);
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

        let mut iterations = 0;
        let mut log_probability = std::f64::NEG_INFINITY;
        let mut old_log_probability = std::f64::NEG_INFINITY;

        while iterations < min_iterations || (log_probability - old_log_probability).abs() > improvement_threshold && iterations < max_iterations {
            old_log_probability = log_probability;

            model.populate_alpha_matrix_and_scale_factors(&observations);
            model.populate_beta_matrix(&observations);
            model.compute_gamma_matrix_and_di_gamma_tensor(&observations);

            model.initial_state_distribution_vector = model.gamma_matrix[0].clone();
            for i in 0..number_of_hidden_state_symbols {
                for j in 0..number_of_hidden_state_symbols {
                    let numerator = model.di_gamma_tensor[..model.di_gamma_tensor.len() - 1]
                        .iter()
                        .map(|di_gamma_matrix| di_gamma_matrix[i][j])
                        .sum::<f64>();
                    let denominator = model.gamma_matrix[..model.gamma_matrix.len() - 1]
                        .iter()
                        .map(|gamma_box| gamma_box[i])
                        .sum::<f64>();

                    model.state_transition_matrix[i][j] = numerator / denominator;
                }

                for j in 0..number_of_observation_symbols {
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;
                    for observation_index in 0..(observations.len() - 1) {
                        if observations[observation_index] == j {
                            numerator += model.gamma_matrix[observation_index][i];
                        }
                        denominator += model.gamma_matrix[observation_index][i];
                    }

                    model.observation_probability_matrix[i][j] = numerator / denominator;
                }
            }

            log_probability = -(model.scale_factors.iter().map(|&scalar| scalar.log2())).sum::<f64>();
            println!("Finished iteration {:?}. New log probability: {:.*}", iterations, 2, log_probability);
            iterations += 1;
        }

        println!("Done training.");

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

    fn populate_alpha_matrix_and_scale_factors(&mut self, observations: &Box<[usize]>) {
        self.scale_factors[0] = 0.0;
        for i in 0..self.get_number_of_hidden_state_symbols() {
            // alpha_0(i) = pi_i * b_i(O_0)
            self.alpha_matrix[0][i] = self.initial_state_distribution_vector[i] * self.observation_probability_matrix[i][observations[0]];
            self.scale_factors[0] += self.alpha_matrix[0][i]
        }

        self.scale_factors[0] = 1.0 / self.scale_factors[0];
        self.alpha_matrix[0] = self.alpha_matrix[0].iter().map(|&alpha_value| self.scale_factors[0] * alpha_value).collect();

        for observation_index in 1..observations.len() {
            self.scale_factors[observation_index] = 0.0;
            for i in 0..self.get_number_of_hidden_state_symbols() {
                // += alpha_t-1(j) * a_ji
                self.alpha_matrix[observation_index][i] = self.alpha_matrix[observation_index - 1]
                    .iter()
                    .enumerate()
                    .map(|alpha_value_pair| alpha_value_pair.1 * self.state_transition_matrix[alpha_value_pair.0][i])
                    .sum::<f64>();

                // = sum(alpha_t-1(j) * a_ji) * b_i(O_t)
                self.alpha_matrix[observation_index][i] *= self.observation_probability_matrix[i][observations[observation_index]];
                self.scale_factors[observation_index] += self.alpha_matrix[observation_index][i];
            }

            self.scale_factors[observation_index] = 1.0 / self.scale_factors[observation_index];
            for i in 0..self.get_number_of_hidden_state_symbols() {
                self.alpha_matrix[observation_index][i] *= self.scale_factors[observation_index];
            }
        }
    }

    fn populate_beta_matrix(&mut self, observations: &Box<[usize]>) {
        // beta_T-1(i) = c_T-1
        self.beta_matrix[observations.len() - 1] = vec![
            self.scale_factors[observations.len() - 1];
            self.get_number_of_hidden_state_symbols()
        ].into_boxed_slice();

        // From T-2 to 0
        for observation_index in (0..(observations.len() - 1)).rev() {
            for i in 0..self.get_number_of_hidden_state_symbols() {
                self.beta_matrix[observation_index][i] = 0.0;
                for j in 0..self.get_number_of_hidden_state_symbols() {
                    // += a_ij * b_j(O_t+1) * beta_t+1(j)
                    self.beta_matrix[observation_index][i] += self.state_transition_matrix[i][j] * self.observation_probability_matrix[j][observations[observation_index + 1]] * self.beta_matrix[observation_index + 1][j];
                }

                self.beta_matrix[observation_index][i] *= self.scale_factors[observation_index];
            }
        }
    }

    fn compute_gamma_matrix_and_di_gamma_tensor(&mut self, observations: &Box<[usize]>) {
        // From 0 to T-2
        for observation_index in 0..(observations.len() - 1) {
            let mut denominator = 0.0;
            for i in 0..self.get_number_of_hidden_state_symbols() {
                for j in 0..self.get_number_of_hidden_state_symbols() {
                    // += alpha_t(t) * a_ij * b_j(O_t+1) * beta_t+1(j)
                    denominator += self.alpha_matrix[observation_index][i] * self.state_transition_matrix[i][j] * self.observation_probability_matrix[j][observations[observation_index + 1]] * self.beta_matrix[observation_index + 1][j];
                }
            }

            for i in 0..self.get_number_of_hidden_state_symbols() {
                self.gamma_matrix[observation_index][i] = 0.0;
                for j in 0..self.get_number_of_hidden_state_symbols() {
                    // += (alpha_t(i) * a_ij * b_j(O_t+1) * beta_t+1(j)) / denom
                    self.di_gamma_tensor[observation_index][i][j] = (self.alpha_matrix[observation_index][i] * self.state_transition_matrix[i][j] * self.observation_probability_matrix[j][observations[observation_index + 1]] * self.beta_matrix[observation_index + 1][j]) / denominator;
                    // += di-gamma_t(i, j)
                    self.gamma_matrix[observation_index][i] += self.di_gamma_tensor[observation_index][i][j];
                }
            }
        }

        let denominator = self.alpha_matrix[observations.len() - 1].iter().sum::<f64>();
        self.gamma_matrix[observations.len() - 1] = self.alpha_matrix[observations.len() - 1].iter().map(|&alpha_value| alpha_value / denominator).collect();
    }
}

fn main() {
    let number_of_observation_symbols = 27;

    let args: Box<[String]> = env::args().collect();
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

    // a, b, c, ..., z, SPACE => 0, 1, 2, ..., 25, 26
    let observations: Box<[usize]> = sanitized_input.chars().map(|ch| match ch {
        ' ' => 26 as usize,
        _ => ch as usize - 'a' as usize,
    }).collect();

    println!("observations length: {:?}", observations.len());

    let time_before_training = SystemTime::now();
    let model = HmmModel::train_model(number_of_hidden_state_symbols, number_of_observation_symbols, &observations);
    println!("Total training time: {:.*}s", 3, time_before_training.elapsed().unwrap().as_millis() as f64 / 1000.0);

    for j in 0..number_of_observation_symbols {
        match j {
            26 => print!("SPACE    "),
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